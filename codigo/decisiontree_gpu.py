import cupy as cp
import numpy as np
import collections
import time

# Con esto todas las operaciones de dentro de la función
# se hacen en el mismo kernel
@cp.fuse
def finish_criteria(t_i, t_d, n_i, n_d):
    return t_i * (n_i - t_i) / n_i + t_d * (n_d - t_d) / n_d

def train_tree(dataset, max_depth=6, min_samples_per_node=1):
    """
    Entrena un árbol con nuestro algoritmo para GPU basado en CUDT.
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
    a = cp.empty((d, 3, N), dtype=cp.float32)
    arbol = np.empty(2*(2**max_depth), dtype=cp.float32)

    #streams = [cp.stream.Stream]
    for i in range(d):
        # El primer campo es el índice de la muestra
        a[i,0] = cp.arange(N, dtype=cp.float32)
        # El segundo los valores para cada atributo
        a[i,1] = cp.array(dataset[:,i], dtype=cp.float32)
        # El tercero las etiquetas
        a[i,2] = cp.array(dataset[:,-1], dtype=cp.float32)
        order = cp.argsort(a[i,1])
        
        # Ordenamos la lista de atributos
        a[i,0] = a[i,0,order]
        a[i,1] = a[i,1,order]
        a[i,2] = a[i,2,order]
        
    # 2. Evaluar nodos.
    ActiveNode = collections.namedtuple('ActiveNode', 'idx start end')
    start_node = ActiveNode(idx=0, start=0, end=N)
    active_list = [start_node]
    
    # Inicio del nivel
    for current_depth in range(max_depth+1):
        next_active_list = []
        best_flag = cp.ones(N, dtype=np.bool)
        if current_depth == 0:
            level_offset = 0 
        else:
            level_offset += 2* (2**(current_depth-1))

        # Recorremos los nodos activos
        for node in active_list:
            n = node.end-node.start
            if n == 1:
                arbol[level_offset+2*node.idx] = -1
                arbol[level_offset+2*node.idx+1] = a[0,2,node.start]
                continue
            elif n <= min_samples_per_node or current_depth == max_depth - 1:
                total = cp.sum(a[0,2,node.start:node.end])
                label = 0 if total/n <= 0.5 else 1
                arbol[level_offset+2*node.idx] = -1
                arbol[level_offset+2*node.idx+1] = label
                continue
            else:
                # Cálculo del criterio de Gini
                scans = cp.cumsum(a[:,2,node.start:node.end], axis=1)
                t_i = scans
                t_d = (-t_i.T + t_i[:,-1]).T
                n_i = cp.arange(1,n+1, dtype=np.int32)
                n_d = cp.arange(n-1,-1,-1,dtype=np.int32)
                
                criteria = finish_criteria(t_i, t_d, n_i, n_d)
    
                cp.place(criteria, cp.diff((a[:,1,node.start:node.end]),append=0)==0, n)
                criteria[cp.isnan(criteria)] = n
                
                # Encontrar mejor punto de corte.
                split_point=cp.argmin(criteria)
                my_list = split_point // n
                my_index = split_point % n
                my_criteria = criteria[my_list, my_index]
                
                if my_criteria == 0:
                    if scans[0,-1] == 0:
                        arbol[level_offset+2*node.idx] = -1
                        arbol[level_offset+2*node.idx+1] = 0
                        continue
                    elif scans[0,-1] == n:
                        arbol[level_offset+2*node.idx] = -1
                        arbol[level_offset+2*node.idx+1] = 1
                        continue
                        
                # Añadimos el nodo de decisión al árbol.
                value = (a[my_list, 1, node.start+my_index] + a[my_list, 1, node.start+my_index + 1])/ 2
                arbol[level_offset+2*node.idx] = my_list
                arbol[level_offset+2*node.idx+1] = value
                # Añadimos a la lista de pendientes los nuevos nodos generados.
                my_index = my_index.get()
                left_node = ActiveNode(idx=2*node.idx, start=node.start, end=node.start+my_index+1)
                right_node = ActiveNode(idx=2*node.idx+1, start=node.start+my_index+1, end=node.end)
                
                next_active_list.append(left_node)
                next_active_list.append(right_node)
                
                # Ponemos a False (0) en Best Flag los atributos que quedan reorganizados a la izquierda.
                best_flag[cp.array(a[my_list,0,node.start:node.start+my_index+1], dtype=cp.int32)] = 0
        
    
        if current_depth == max_depth - 1 or len(next_active_list) == 0:
            return arbol
        
        # Reorganizar listas de atributos [Una vez por nivel de profundidad]
        my_flag = best_flag[cp.array(a[:,0], dtype=cp.int32)]
        buffer = cp.logical_not(my_flag)
        address = cp.tile(cp.arange(N),d).reshape(d,N)

        for node in active_list:
            s = node.start
            e = node.end
            n = node.end - node.start
            false_array = cp.cumsum(buffer[:,s:e], axis=1)
            my_buffer = (-false_array.T + false_array[:,-1]).T + cp.arange(0,n)
            address[:, s:e] = false_array - 1
            for i in range(d):
                address[i, s:e][my_flag[i,s:e]] = my_buffer[i][my_flag[i,s:e]]
            address[:, s:e] += s
            
        for i in range(d):
            aux = cp.copy(a[i,0])
            a[i,0, address[i]] = aux
            aux = cp.copy(a[i,1])
            a[i,1, address[i]] = aux
            aux = cp.copy(a[i,2])
            a[i,2, address[i]] = aux
            
            
        active_list = next_active_list
        


def predict(sample, tree):
    # 1. Empiezo en el nodo raíz del arbol
    depth = 0
    i = 0
    my_attr = tree[0]
    level_offset = 0
    # 2. Me muevo por el árbol mientras que esté en un nodo de decisión
    while my_attr != -1: 
        value = tree[level_offset+2*i+1]
        # Evalúo la condición
        if sample[int(my_attr)] <= value:
        # En este caso en el siguiente nivel voy al nodo de la izquierda
            i = 2 * i
        else:
            # En el otro caso me voy al nodo de la derecha
            i = 2 * i + 1
        depth += 1
        level_offset += 2* (2**(depth-1))
        my_attr = tree[level_offset+2*i]
        
    return int(tree[level_offset+2*i+1])


def evaluar(dataset, modelo):
    aciertos = 0
    for sample in dataset:
        etiqueta = predict(sample, modelo)
        if int(etiqueta) == int(sample[-1]):
            aciertos += 1
        
    return aciertos / dataset.shape[0]

def validacion_cruzada(dataset, k=10, depth=5, my_min=1, seed=None):
    np.random.seed(seed)
    n = dataset.shape[0]
    tam = n // k
    samples_test = []

    
    x = np.arange(n)
    for i in range(k):
        samples_test.append(np.random.choice(x, tam, replace=False))
        
                
    times = np.zeros(k, np.float32)
    accuracy = np.zeros(k, np.float32)
    all_indexes = np.arange(n)
    for i in range(k):
        samples_train = np.delete(all_indexes, samples_test[i])
        inicio = time.time()
        model = train_tree(dataset[samples_train], depth, my_min)
        fin = time.time()
        times[i] = fin-inicio
        accuracy[i] = evaluar(dataset[samples_test[i]], model)

        
    return [np.mean(accuracy), np.mean(times)]


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
    arbol = np.empty((2*(2**max_depth)), dtype=np.float32)
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
    ActiveNode = collections.namedtuple('ActiveNode', 'idx start end')
    start_node = ActiveNode(idx=0, start=0, end=N)
    active_list = [start_node]
    
    # Inicio del nivel
    for current_depth in range(max_depth+1):
        next_active_list = []
        best_flag = np.ones(N, dtype=np.bool)
        if current_depth == 0:
            level_offset = 0 
        else:
            level_offset += 2* (2**(current_depth-1))
        
        inicio = time.time()
        # Recorremos los nodos activos
        for node in active_list:
            n = node.end-node.start
            if n == 1:
                arbol[level_offset+2*node.idx] = -1
                arbol[level_offset+2*node.idx+1] = a[0,2,node.start]
                continue
            elif n <= min_samples_per_node or current_depth == max_depth - 1:
                total = np.sum(a[0,2,node.start:node.end])
                label = 0 if total/n <= 0.5 else 1
                arbol[level_offset+2*node.idx] = -1
                arbol[level_offset+2*node.idx+1] = label
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
                        arbol[level_offset+2*node.idx] = -1
                        arbol[level_offset+2*node.idx+1] = 0
                        continue
                    elif scans[0,-1] == n:
                        arbol[level_offset+2*node.idx] = -1
                        arbol[level_offset+2*node.idx+1] = 1
                        continue
                        
                # Añadimos el nodo de decisión al árbol.
                arbol[level_offset+2*node.idx] = my_list
                arbol[level_offset+2*node.idx+1] = value
                
                # Añadimos a la lista de pendientes los nuevos nodos generados.
                left_node = ActiveNode(idx=2*node.idx, start=node.start, end=node.start+my_index+1)
                right_node = ActiveNode(idx=2*node.idx+1, start=node.start+my_index+1, end=node.end)
                
                next_active_list.append(left_node)
                next_active_list.append(right_node)
                
                # Ponemos a False (0) en Best Flag los atributos que quedan reorganizados a la izquierda.
                best_flag[np.array(a[my_list,0,node.start:node.start+my_index+1], dtype=np.int32)] = 0
        
        fin = time.time()
        if current_depth == max_depth - 1 or len(next_active_list) == 0:
            return arbol
        
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
        
    return arbol


def validacion_cruzada_np(dataset, k=10, depth=5, my_min=1, seed=None):
    np.random.seed(seed)
    n = dataset.shape[0]
    tam = n // k
    samples_test = []

    x = np.arange(n)
    for i in range(k):
        samples_test.append(np.random.choice(x, tam, replace=False))
        
    times = np.zeros(k, np.float32)
    accuracy = np.zeros(k, np.float32)
    all_indexes = np.arange(n)
    for i in range(k):
        samples_train = np.delete(all_indexes, samples_test[i])
        inicio = time.time()
        model = train_tree_np(dataset[samples_train], depth, my_min)
        fin = time.time()
        times[i] = fin-inicio
        accuracy[i] = evaluar(dataset[samples_test[i]], model)

    return [np.mean(accuracy), np.mean(times)]

if __name__ == '__main__':
    magic = np.genfromtxt('../datasets/magic04.data', delimiter=',', dtype=np.float32)
    arbol = train_tree(magic,9)