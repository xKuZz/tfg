import numpy as np
import cupy as cp
import math
from numba import cuda, int32, float32, void
import numba
import time
import utils
import collections


@cuda.jit
def inplace_reorder(array, indexes, buffer):
    """
    Kernel utilizado para recolocar elementos que ya están en
    la memoria del dispositivo  (array = array[indexes])
    :param array Array a reorganizar.
    :indexes Índices para reodernar el array.
    :buffer Búfer para almacenar datos temporalmente.
    """
    idx = cuda.grid(1)
    if idx < indexes.size:
        buffer[idx] = array[indexes[idx]]
        cuda.syncthreads()
        array[idx] = buffer[idx]


@cuda.jit
def split_criteria(scan, criteria, values, n, total):
    """
    Kernel utilizado para calcular el criterio de Gini
    para un nodo de nuestro árbol de decisión.
    :param scan Array con una operación de scan exclusivo en
           función de la etiqueta de cada muestra para todas
           las listas de atributos.
    :param criteria Array que contendrá el valor del crierio
           de Gini para la subdivisión asociada a esa posición.
    :param values Array con los valores de cada muestra del nodo
           para todas las listas de atributos.
    :param n Entero que representa el número de muestras del nodo.
    :param total Array de 1 elemento en el dispositivo procedente
           de la operación de scan exclusivo que contiene la suma
           total de elementos de la clase positiva (1).
    """
    shared_aux = cuda.shared.array(2, numba.int32)
    if cuda.threadIdx.x == 0:
        shared_aux[0] = n
        shared_aux[1] = total[0]
    cuda.syncthreads()
    
    idx = cuda.grid(1)
    if idx < scan.size:
        n_i = idx % shared_aux[0] + 1
        n_d = shared_aux[0] - n_i
        if n_d != 0 and values[idx] != values[idx + 1]:
            t_i = scan[idx+1]
            t_d = shared_aux[1] - t_i
            criteria[idx] = (t_i * (n_i - t_i)/(n_i)) + (t_d * (n_d - t_d)/(n_d))
        else:
            criteria[idx] = shared_aux[0]
            
            
@cuda.jit
def set_best_flag(flag, indexes):
    """
    Kernel que pone a 0 ciertas posiciones específicas en un array.
    :param flag Array a modificar.
    :param indexes Posiciones a modificar.
    """
    idx = cuda.grid(1)
    if idx < indexes.size:
        flag[indexes[idx]] = 0
        

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
    idx = cuda.grid(1)
    if idx < indexes.size:
        my_flag[idx] = best_flag[indexes[idx]]
        buffer[idx] = not my_flag[idx]
        
@cuda.jit
def calc_addresses(my_flag, false_counter, total_false, address, my_offset):
    """
    Kernel para calcular las posiciones correspondientes
    a los cambios necesarios para evitar reodenar la
    lista de atributos en un nodo.
    :param my_flag Array de booleanos que indica si pertenece a la
                   subdivisión de la izquierda o la derecha.
    :param false_counter Scan exclusivo para contar los elementos 0 de my_flag.
    :param total_false Número total de elementos 0 de my_flag.
    :param address Posición que ocupará el elemento para estar ordenado dentro
           de su nodo.
    :param my_offset Posición inicial del nodo a evaluar.
    """
    idx = cuda.grid(1)
    if idx < my_flag.size:
        if idx == my_flag.size - 1:
            address[idx] = idx if my_flag[idx] else total_false[0] - 1
        else:
            address[idx] = idx + total_false[0] - false_counter[idx+1] if my_flag[idx] else false_counter[idx+1] - 1
        address[idx] += my_offset
    
@cuda.jit
def finish_split(address, rids, values, labels, buffer_int, buffer_float, buffer_int2):
    """
    Kernel que realiza la reorganización de la lista
    de atributos de todos los nodos en función de las
    posiciones calculadas
    :param address Nuevas posiciones.
    :param rids Campo de la lista de atributos con el ID de muestra.
    :param values Campo de la lista de atributos con el valor asociado.
    :param labels Campo de la lista de atributos con la etiqueta asociada.
    :param buffer_int Búfer auxiliar para enteros.
    :param buffer_float Búfer auxiliar para flotantes.
    :param buffer_int2 Búfer auxiliar para enteros.
    """
    idx = cuda.grid(1)
    
    if idx < address.size:
        buffer_int[idx] = rids[idx]
        buffer_float[idx] = values[idx]
        buffer_int2[idx] = labels[idx]
    cuda.syncthreads()
    
    if idx < address.size:
        rids[address[idx]] = buffer_int[idx]
        values[address[idx]] = buffer_float[idx]
        labels[address[idx]] = buffer_int2[idx]

@cuda.jit
def set_range(array):
    """
    Kernel que escribe el rango [0,1,2,...,n] en un
    array del dispositivo de N elementos.
    :param array Array sobre el que escribir.
    """
    idx = cuda.grid(1)
    if idx < array.size:
        array[idx] = idx

@cuda.jit
def set_to_one(array):
    """
    Kernel que pone 1 todos los elementos
    de un array del dispositivo.
    :param array Array sobre el que escribir.
    """
    idx = cuda.grid(1)
    if idx < array.size:
        array[idx] = 1
        

def train_tree(dataset, max_depth=6, min_samples_per_node=1, tpb=128):
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
    N, d = dataset.shape
    d -= 1
    buffer_int = cuda.device_array(N, dtype=np.int32)
    buffer_float = cuda.device_array(N, dtype=np.float32)
    
    # 1.1 Generamos las listas de atributos y las reodenamos en orden
    #     ascendente de valor
    attribute_lists = []
    indexes = np.array(np.arange(N), dtype=np.int32)
    blocks = N // tpb + 1
    for i in range(d):
        my_list = []
        y = cuda.to_device(np.ascontiguousarray(dataset[:,i])) 
        order = np.array(cp.argsort(cp.array(y)).get(), dtype=np.int32)
        inplace_reorder[blocks, tpb](y, order, buffer_float)
        
        x = cuda.to_device(indexes)
        inplace_reorder[blocks, tpb](x, order, buffer_int)
        z = cuda.to_device(np.ascontiguousarray(dataset[:,-1], np.int32))
        inplace_reorder[blocks, tpb](z, order, buffer_int)
        my_list.append(x)
        my_list.append(y)
        my_list.append(z)
        attribute_lists.append(my_list)
        
    # 1.2 Generamos el nodo inicial
    outputs = []
    ActiveNode = collections.namedtuple('ActiveNode', 'idx start end')
    
    start_node = ActiveNode(idx=0, start=0, end=N)
    active_list = [start_node]
    
    # 1.3 Generamos una única vez los arrays del dispositivo
    #     que reutilizaremos según sea conveniente.
    scan = cuda.device_array(N*d, dtype=np.float32)
    criteria = cuda.device_array(N*d, np.float32)
    values = cuda.device_array(N*d, np.float32)
    best_flag = cuda.device_array(N, np.bool)
    my_flag = cuda.device_array(N, np.int32)
    address = cuda.device_array(N, np.int32)
    
    # 2. Recorremos los niveles de profundidad
    for current_depth in range(max_depth):
        level = {}
        next_active_list = []
        set_to_one[N//tpb + 1, tpb](best_flag)
        
        # 2.1 Buscamos split points
        for node in active_list:
            n = node.end - node.start
            # Criterio de Poda: Mínimo de elmentos en hoja o último nivel de profundidad.
            if n == 1:
                level[node.idx] = (False, attribute_lists[i][2][node.start])
                continue
            elif n <= min_samples_per_node or current_depth == max_depth - 1:
                my_total = utils.reduce_sum(attribute_lists[i][2][node.start:node.end])
                label = 0 if my_total/n <= 0.5 else 1
                level[node.idx] = (False, label)
                continue
            else:
                # Recorremos la lista de atributos.
                offset = 0
                # Realizamos el scan.
                for i in range(d):
                    values[offset:offset+n] = attribute_lists[i][1][node.start:node.end]
                    aux = attribute_lists[i][2][node.start:node.end].copy_to_host()
                    total = utils.scan(attribute_lists[i][2][node.start:node.end], scan[offset:offset+n])
                    offset += n
               
                if total[0] == 0 or total[0] == n:
                    level[node.idx] = (False, total[0])
                    continue
                    
                # Calculamos el criterio de Gini.
                split_criteria[d * n // tpb + 1, tpb](scan[:d*n], criteria[:d*n], values[:d*n], n, total)
                
                # Calculamos la reducción.
                min_index = utils.reduce_min_index(criteria[:d*n])
                my_attr_list = min_index // n
                my_index = min_index % n


                # Añadimos el nuevo nodo a la salida del árbol.
                the_value = (values[min_index] + values[min_index + 1]) / 2
                level[node.idx] = (True, my_attr_list, my_index, criteria[min_index]*2/N, the_value)

                # Añadimos a la lista de pendientes los nuevos nodos generados.
                left_node = ActiveNode(idx=2*node.idx, start=node.start, end=node.start+my_index+1)
                right_node = ActiveNode(idx=2*node.idx+1, start=node.start+my_index+1, end=node.end)

                next_active_list.append(left_node)
                next_active_list.append(right_node)

                # Ponemos a False (0) en Best Flag los atributos que quedan reorganizados a la izquierda.
                set_best_flag[n//tpb+1, tpb](best_flag, attribute_lists[my_attr_list][0][left_node.start:left_node.end])

        # Añadimos el nivel del árbol a la salida
        outputs.append(level)
            
        if current_depth == max_depth - 1:
            return outputs
        
        # 2.2 Reorganizamos las listas de atributos
        for i in range(d):
            set_range[N//tpb+1, tpb](address)
            for node in active_list:
                s = node.start
                e = node.end
                n = e - s
                if n > 1 and i != level[node.idx][1]:
                    set_my_flag[n // tpb + 1, tpb](my_flag[s:e], buffer_int[s:e], best_flag, attribute_lists[i][0][s:e])
                    total_false = utils.scan(buffer_int[s:e], scan[s:e])                                         
                    calc_addresses[n // tpb + 1, tpb](my_flag[s:e], scan[s:e], total_false, address[s:e], s)
            
            finish_split[N//tpb+1, tpb](address, attribute_lists[i][0], attribute_lists[i][1], attribute_lists[i][2],
                                        buffer_int, buffer_float, scan)
            
        # 2.3 Cambiamos la lista de nodos activos a la del siguiente nivel
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