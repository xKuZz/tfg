import numpy as np
import cupy as cp
import math
from numba import cuda, int32, float32, void
import numba
import time
import utils
import collections

@cuda.jit
def fill_buffer(array, array2):
    row = cuda.blockIdx.y
    col = cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x
    if col < array.shape[1]:
        array[row, col] = array2[row, col]

@cuda.jit
def fill_2d(array, indexes, array2):
    idx = cuda.grid(1)
    ncols = array.shape[1]
    row = idx // ncols
    col = idx % ncols

    if idx < array.size:
        array[row, col] = array2[row, indexes[row,col]]

@cuda.jit
def fill_2d_b(array, indexes, array2):
    idx = cuda.grid(1)
    ncols = array.shape[1]
    row = idx // ncols
    col = idx % ncols

    if idx < array.size:
        array[row, indexes[row,col]] = array2[row, col]


@cuda.jit
def fill_2d_label(array, indexes, array2):
    idx = cuda.grid(1)
    ncols = array.shape[1]
    row = idx // ncols
    col = idx % ncols

    if idx < array.size:
        array[row, col] = array2[indexes[row,col]]


          
@cuda.jit
def set_best_flag(flag, indexes):
    """
    Kernel que pone a 0 ciertas posiciones específicas en un array.
    :param flag Array a modificar.
    :param indexes Posiciones a modificar.
    """
    idx = cuda.grid(1)
    if idx < indexes.size:
        flag[indexes[idx]] = False
        

@cuda.jit
def set_my_flag(my_flag, buffer, best_flag, indexes):
    """
    Kernel que toma los valores booleanos de otros arrays
    y también rellena un array auxiliar con su opuesto
    my_flag = best_flag[indexes]
    buffer = not my_flag
    :param my_flag Array a rellenar con los booleanos.
    :param buffer Array a rellenar con los opuestos.
    :param best_flag Array del que tomar los índices de los
           booleanos.
    :param indexes Índices para acceder al array de booleanos.
    """
    row = cuda.blockIdx.y
    col = cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x
    if col < my_flag.shape[1]:
        data = best_flag[indexes[row, col]]
        my_flag[row, col] = data
        buffer[row, col] = not data
        

@cuda.jit
def set_range(array):
    """
    Kernel que escribe el rango [0,1,2,...,n] en un
    array del dispositivo de N elementos.
    :param array Array sobre el que escribir.
    """
    row = cuda.blockIdx.y
    col = cuda.threadIdx.x+cuda.blockIdx.x*cuda.blockDim.x
    if col < array.shape[1]:
        array[row, col] = col



def train_tree(dataset, max_depth=6, min_samples_per_node=1, tpb=1024, nstreams=3):
    """
    Entrena un árbol con nuestro algoritmo para GPU basado en CUDT.
    :param dataset Conjunto de datos con el que entrenar el árbol.
    :param max_depth Profundida máxima del árbol.
    :param min_samples_per_node Mínimo de elementos en un nodo para
           seguir siendo evaluado.
    :param tpb Hebras por bloque CUDA.
    :pre dataset ha de ser un array de Numpy con todas las variables
         de tipo np.float32 y la última columna las etiquetas. Dichas
         etiquetas han de corresponderse a una clasificación binaria [0,1].
         tpb ha de ser un valor válido para bloques unidimensionales de CUDA.
    """

    """
    def buffer_reorder(data, order, my_buffer, stream=0):
    fill_idx[blocks, tpb, stream](data, order, my_buffer)
    cuda_replace[blocks,tpb, stream](data, my_buffer)
    return data
    """

    N, d = dataset.shape
    d -= 1
    
    values = cuda.device_array((d, N), dtype=np.float32)
    labels = cuda.device_array((d, N), dtype=np.int32)
    # 1.1 Generamos las listas de atributos y las ponemos en orden
    #     ascendente de valor
    
    blocks = N // tpb + 1
    dataset = np.ascontiguousarray(dataset.T)
    
    streams = [cuda.stream() for i in range(3)]

    d_scan = cuda.device_array((d, N), np.float32, stream=streams[2])
    
    best_flag = cuda.device_array(N, np.bool, stream=streams[0])
    my_flag = cuda.device_array((d,N), np.bool, stream=streams[1])
    buffer_int = cuda.device_array((d, N), np.int32, stream=streams[2])
    buffer_int2 = cuda.device_array((d, N), np.int32, stream=streams[0])

    locks = [cuda.device_array(1, np.int32, stream=stream) for stream in streams]
    locks[:][0] = 0
    my_mins = [cuda.device_array(1, np.float32, stream=stream) for stream in streams]
    my_min_idxs = [cuda.device_array(2, np.int32, stream=stream) for stream in streams]
    my_totals = [cuda.device_array(1, dtype=np.int32, stream=stream) for stream in streams]

    address = cuda.device_array((d, N), np.int32, stream=streams[2])
    set_range[(N//tpb+1, d), tpb, streams[2]](address)
    with cuda.pinned(dataset):
        d_labels = cuda.to_device(dataset[-1], stream=streams[0])
        d_dataset = cuda.to_device(dataset[:-1], stream=streams[1])
       
        indexes = cp.argsort(cp.array(dataset[:-1]), axis=1)
        indexes = cuda.to_device(indexes, stream=0)
        
        fill_2d[d*N//tpb+1, tpb, streams[0]](values, indexes, d_dataset)
        fill_2d_label[d*N//tpb+1, tpb, streams[1]](labels, indexes, d_labels)
        
        
    cuda.synchronize()
   
    
    # 1.2 Generamos el nodo inicial
    outputs = []
    ActiveNode = collections.namedtuple('ActiveNode', 'idx start end')
    
    start_node = ActiveNode(idx=0, start=0, end=N)
    active_list = [start_node]
    
    # 2. Recorremos los niveles de profundidad
    for current_depth in range(max_depth):
        best_flag[:] = True
        level = {}
        next_active_list = []

        
        # 2.1 Buscamos split points
        for i, node in enumerate(active_list):
            n = node.end - node.start
            s = node.start
            e = node.end
            node_tpb = min(max(32, 2**math.ceil(math.log2(n))), tpb)
            id_stream = i % 3
            my_stream = streams[id_stream]
            # Criterio de Poda: Mínimo de elementos en hoja o último nivel de profundidad.
            if n == 1:
                level[node.idx] = (False, values[0, node.start])
                continue
            elif n <= min_samples_per_node or current_depth == max_depth - 1:
                my_totals[id_stream][0] = 0
                utils.warp_based_reduce_sum[n//node_tpb+1, node_tpb, my_stream](labels[0,node.start:node.end], my_totals[id_stream])
                my_total = my_totals[id_stream].copy_to_host(stream=my_stream)[0]
                label = 0 if my_total/n <= 0.5 else 1
                level[node.idx] = (False, label)
                continue
            else:
                # Realizamos el scan de los labels
                aux = cuda.device_array((d, n//node_tpb+1), dtype=np.float32, stream=my_stream)
                aux[:] = 0
      
                my_total = utils.multi_scan_with_gini(labels[:,s:e] ,d_scan[:,s:e], values[:,s:e], my_mins[id_stream], aux, node_tpb, my_stream)
                if my_total == 0 or my_total == n:
                    level[node.idx] = (False, my_total)
                    aux[:,:] = 0
                    continue

                blocks = (n//node_tpb+1, d)
                utils.min_index_reduction[blocks, node_tpb, my_stream](d_scan[:,s:e], my_mins[id_stream], my_min_idxs[id_stream], aux, locks[id_stream])
                my_host_idx = my_min_idxs[id_stream].copy_to_host(stream=my_stream)        
                my_attr_list = my_host_idx[0]
                my_index = my_host_idx[1]
                
                # Ponemos a False (0) en Best Flag los atributos que quedan reorganizados a la izquierda.
                set_best_flag[n//node_tpb+1, node_tpb, my_stream](best_flag, indexes[my_attr_list, node.start:node.start+my_index+1])
                set_my_flag[(n // tpb + 1, d), tpb, my_stream](my_flag[:,s:e], buffer_int[:,s:e], best_flag, indexes[:,s:e])
                utils.multi_scan_with_address(buffer_int[:,s:e], d_scan[:,s:e], address[:,s:e], my_flag[:,s:e], aux, node.start, node_tpb, my_stream)    

                # Añadimos el nuevo nodo a la salida del árbol.
                my_values = values[my_attr_list, node.start+my_index:node.start+my_index+2].copy_to_host(stream=my_stream)
                the_value = (my_values[0] + my_values[1])/2
                level[node.idx] = (True, my_attr_list, my_index, the_value)

                # Añadimos a la lista de pendientes los nuevos nodos generados.
                left_node = ActiveNode(idx=2*node.idx, start=node.start, end=node.start+my_index+1)
                right_node = ActiveNode(idx=2*node.idx+1, start=node.start+my_index+1, end=node.end)

                next_active_list.append(left_node)
                next_active_list.append(right_node)

        
        # Añadimos el nivel del árbol a la salida
        cuda.synchronize()
        outputs.append(level)
            
        if current_depth == max_depth - 1:
            return outputs
        
        # 2.2 Reorganizamos las listas de atributos
        
        fill_buffer[(N // tpb + 1, d), tpb, streams[0]](d_scan, values)
        fill_buffer[(N // tpb + 1, d), tpb, streams[1]](buffer_int, indexes)
        fill_buffer[(N // tpb + 1, d), tpb, streams[2]](buffer_int2, labels)
        fill_2d_b[d*N//tpb+1, tpb, streams[0]](values, address, d_scan)
        fill_2d_b[d*N//tpb+1, tpb, streams[1]](indexes, address, buffer_int) 
        
        fill_2d_b[d*N//tpb+1, tpb, streams[2]](labels, address, buffer_int2) 
        
        # 2.3 Cambiamos la lista de nodos activos a la del siguiente nivel
        active_list = next_active_list
        cuda.synchronize()
        
    return outputs


def train_tree_np(dataset, max_depth=6, min_samples_per_node=1):
    """
    Entrena un árbol con nuestro algoritmo pero para CPU.
    :param dataset Conjunto de datos con el que entrenar el árbol.
    :param max_depth Profundida máxima del árbol.
    :param min_samples_per_node Mínimo de elementos en un nodo para
           seguir siendo evaluado.
    
    :pre dataset ha de ser un array de Numpy con todas las variables
         de tipo np.float32 y la última columna las etiquetas. Dichas
         etiquetas han de corresponderse a una clasificación binaria [0,1].
         tpb ha de ser un valor válido para bloques unidimensionales de CUDA.
    """
    # 1. Generar listas de atributos
    N,d = dataset.shape
    d -= 1
    a = np.empty((d, 3, N), dtype=np.float32)
    for i in range(d):
        # El primer campo es el índice de la muestra
        a[i,0] = np.arange(N)
        # El segundo los valores para cada atributo
        a[i,1] = np.array(dataset[:,i], dtype=np.float32)
        # El tercero las etiquetas
        a[i,2] = np.array(dataset[:,-1], dtype=np.float32)
        order = np.argsort(a[i,1])
        
        # Ordenamos la lista de atributos
        a[i,0] = a[i,0,order]
        a[i,1] = a[i,1,order]
        a[i,2] = a[i,2,order]
        
    # 2. Evaluar nodos.
    outputs = []
    ActiveNode = collections.namedtuple('ActiveNode', 'idx start end')
    start_node = ActiveNode(idx=0, start=0, end=N)
    active_list = [start_node]
    
    # Inicio del nivel
    for current_depth in range(max_depth+1):
        level = {}
        next_active_list = []
        best_flag = np.ones(N, dtype=np.bool)
        
        # Recorremos los nodos activos
        for node in active_list:
            n = node.end-node.start
            if n == 1:
                level[node.idx] = (False, int(a[0,2,node.start]))
                continue
            elif n <= min_samples_per_node or current_depth == max_depth - 1:
                total = np.sum(a[0,2,node.start:node.end])
                label = 0 if total/n <= 0.5 else 1
                level[node.idx] = (False, label)
                continue
            else:
                # Cálculo del criterio de Gini
                scans = np.cumsum(a[:,2,node.start:node.end], axis=1)
                t_i = scans
                t_d = (-t_i.T + t_i[:,-1]).T
                n_i = np.arange(1,n+1, dtype=np.int32)
                n_d = np.arange(n-1,-1,-1,dtype=np.int32)
                
                criteria = t_i * (n_i - t_i) / n_i + t_d * (n_d - t_d) / n_d
    
                np.place(criteria, np.diff((a[:,1,node.start:node.end]),append=0)==0, n)
                criteria[np.isnan(criteria)] = n
                
                # Encontrar mejor punto de corte.
                split_point=np.argmin(criteria)
                my_list = split_point // n
                my_index = split_point % n
                my_criteria = criteria[my_list, my_index]
                
                value = (a[my_list, 1, node.start+my_index] + a[my_list, 1, node.start+my_index + 1]) / 2
                if my_criteria == 0:
                    if scans[0,-1] == 0:
                        level[node.idx] = (False, 0)
                        continue
                    elif scans[0,-1] == n:
                        level[node.idx] = (False, 1)
                        continue
                        
                # Añadimos el nodo de decisión al árbol.
                level[node.idx] = (True, my_list, my_index, value)
                
                # Añadimos a la lista de pendientes los nuevos nodos generados.
                left_node = ActiveNode(idx=2*node.idx, start=node.start, end=node.start+my_index+1)
                right_node = ActiveNode(idx=2*node.idx+1, start=node.start+my_index+1, end=node.end)
                
                next_active_list.append(left_node)
                next_active_list.append(right_node)
                
                # Ponemos a False (0) en Best Flag los atributos que quedan reorganizados a la izquierda.
                best_flag[np.array(a[my_list,0,node.start:node.start+my_index+1], dtype=np.int32)] = 0
        
        # Añadimos nivel a la salida
        outputs.append(level)
        
    
        if current_depth == max_depth - 1 or len(next_active_list) == 0:
            return outputs
        
        # Reorganizar listas de atributos
        my_flag = best_flag[np.array(a[:,0], dtype=np.int32)]
        buffer = np.logical_not(my_flag)
        address = np.tile(np.arange(N),d).reshape(d,N)
        for node in active_list:
            s = node.start
            e = node.end
            n = node.end - node.start
            false_array = np.cumsum(buffer[:,s:e], axis=1)
            my_buffer = (-false_array.T + false_array[:,-1]).T + np.arange(0,n)
            address[:, s:e] = false_array - 1
            for i in range(d):
                address[i, s:e][my_flag[i,s:e]] = my_buffer[i][my_flag[i,s:e]]
            address[:, s:e] += s
       
        for i in range(d):
            aux = np.copy(a[i,0])
            aux_2 = np.copy(a[i,1])
            aux_3 = np.copy(a[i,2])
            a[i,0, address[i]] = aux
            a[i,1, address[i]] = aux_2
            a[i,2, address[i]] = aux_3
        active_list = next_active_list
        
    return outputs
    
def predict(sample, tree):
    # 1. Empiezo en el nodo raíz del arbol
    depth = 0
    i = 0
    my_tuple = tree[depth][i]
    # 2. Me muevo por el árbol mientras que esté en un nodo de decisión
    while my_tuple[0]: 
        # Evalúo la condición
        if sample[my_tuple[1]] <= my_tuple[-1]:
            # En este caso en el siguiente nivel voy al nodo de la izquierda
            i = 2 * i
        else:
            # En el otro caso me voy al nodo de la derecha
            i = 2 * i + 1
        depth += 1
        my_tuple = tree[depth][i]
        
    return my_tuple[1]


def evaluar(dataset, modelo):
    aciertos = 0
    for sample in dataset:
        etiqueta = predict(sample, modelo)
        if int(etiqueta) == int(sample[-1]):
            aciertos += 1
        
    return aciertos / dataset.shape[0]

if __name__ == '__main__':

    #dataset = np.genfromtxt('../datasets/miniboone_PID.txt', dtype=np.float32)
    #labels = np.zeros((dataset.shape[0],1))
    #labels[:36499] = 1
    #magic = np.hstack((dataset,labels))


    magic = np.genfromtxt('../datasets/magic04.data', delimiter=',', dtype=np.float32)
    inicio = time.time()
    arbol = train_tree(magic)
    fin = time.time()
    
    print('Tiempo', fin-inicio)
    inicio = time.time()
    arbol = train_tree(magic, 4)
    fin = time.time()
    print('Tiempo', fin-inicio)
    print('Precisión', evaluar(magic, arbol))