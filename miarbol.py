"""
Cargar datasets.
Datasets seleccionados:
Spambase
Magic Gamma Telescope
"""
import numpy as np
import cupy as cp
import cmath
from numba import cuda, int32, float32, void
import numba
import time

"""
SPAMBASE
"""
def getdataset_spambase():
    return np.genfromtxt('./datasets/spambase.data', delimiter=',', dtype=np.float32)

"""
Magic Gamma Telescope
"""
def getdataset_magic():
    # 1 es Clase G y 0 es Clase H, ya modificado en el archivo que se lee por comodidad
    return np.genfromtxt('./datasets/magic04.data', delimiter=',', dtype=np.float32)

"""
Fase 1 Generar listas de atributos.
"""
def gen_attribute_lists(dataset):
    attribute_lists = []
    record_idx = cp.arange(len(dataset), dtype=np.float32)
    for feature_idx in range(dataset.shape[1]-1):
        my_list = cp.vstack((dataset[:, feature_idx], dataset[:, -1], record_idx))
        attribute_lists.append(my_list)
    return attribute_lists

"""
Fase 1.2 Ordenar listas de atributos
"""
def cupy_sort_my_attr_lists(attr_lists):
    for i in range(len(attr_lists)):
        indexes = cp.argsort(attr_lists[i][0])
        for j in range(3):
            attr_lists[i][j] = attr_lists[i][j][indexes]
    return attr_lists

"""
Fase 2 Encontrar split points (modificada)
"""

@cuda.jit('int32[:], float32[:], float32[:], int32[:], int32[:]', fastmath=True)
def split_criteria(scan, criteria, values, n_i, n_d):
    idx = cuda.grid(1)
    if idx < values.size:
        if n_d[idx] != 0 and values[idx] != values[idx + 1]:
            t_i = scan[idx]
            t_d = scan[idx + n_d[idx]] - t_i
            criteria[idx] = (t_i * (n_i[idx] - t_i)/(n_i[idx])) + (t_d * (n_d[idx] - t_d)/(n_d[idx]))
        else:
            criteria[idx] = n_i[idx] + n_d[idx]


            
        
        
def cupy_find_split_point(attr_lists, positions,  min_samples_per_node, tpb, last=False):
    n = len(attr_lists[0][0])
    
    # Preparamos los arrays del dispositivo
    scan = cp.empty(n * len(attr_lists), dtype=cp.int32)
    criteria = cp.empty(n * len(attr_lists), dtype=cp.float32)
    values = cp.empty(n * len(attr_lists), dtype=cp.float32)
    n_i = cp.empty(n * len(attr_lists), dtype=cp.int32)
    n_d = cp.empty(n * len(attr_lists), dtype=cp.int32)
    
    output = {}
    # Rellenamos los arrays con los elementos para calcular
    offset = 0
    
    # Recorremos el array de posiciones (nodos activos e inactivos)
    for i in range(len(positions) - 1):
        if positions[i + 1] > 0:
            n = positions[i + 1] - abs(positions[i])
            
            # Criterio de poda de nodo: Menos de un número de muestras o prof máxima.
            if n <= min_samples_per_node or last:
                total = cp.sum(attr_lists[0][1,abs(positions[i]):positions[i+1]])
                label = 1 if total/n > 0.5 else 0
                if last:
                    output[i] = (False,  label, 'Última capa')
                else:
                    output[i] = (False,  label, 'Mínimo elementos hoja')
                    
                # Indicamos que el nodo no está activo y continuamos si procede
                positions[i + 1] = -positions[i + 1]
                offset += n * len(attr_lists)
                continue
            
            my_n_i = cp.arange(1, n+1, dtype=cp.int32)
            my_n_d = cp.arange(n-1, -1, -1,dtype=cp.int32)
            for j in range(len(attr_lists)):
                scan[offset:offset+n] = cp.cumsum(attr_lists[j][1,abs(positions[i]):positions[i+1]])
                
                # Criterio de terminación de nodo: Todas son de la misma etiqueta
                # Basta con comprobarlo en una lista de atributos.
                if j == 0 and (scan[offset+n-1] == n or scan[offset + n-1] == 0):
                    # print('Podando con scan', scan[offset + n-1], 'de', n)
                    # Preparamos la salida
                    label = 1 if scan[offset+n-1] == n else 0
                    output[i] = (False,  label, 'Nodo terminal')
                    
                    # Indicamos que el nodo no está activo y continuamos si procede
                    positions[i + 1] = -positions[i + 1]
                    offset += n * len(attr_lists)
                    break
                        
                values[offset:offset+n] = attr_lists[j][0, abs(positions[i]):positions[i+1]]
                n_i[offset:offset+n] = my_n_i
                n_d[offset:offset+n] = my_n_d
                offset += n
        else:
            offset += (abs(positions[i+1])-abs(positions[i]))*len(attr_lists)

    if not last:
        # Lanzamos el kernel de Numba
        blocks = len(attr_lists[0][0]) * len(attr_lists) // tpb + 1
        split_criteria[blocks, tpb](scan,criteria, values, n_i, n_d)


        # Realizamos la reducción para encontrar el atributo que mejor hace la separación
        # pero sólo en las partes activas
        offset = 0
        d = len(attr_lists)
        for i in range(len(positions) - 1):
            # Comprobamos si es un nodo activo
            if positions[i + 1] > 0:
                my_n = positions[i + 1] - abs(positions[i])
                my_nd = my_n * d

                my_best = int(cp.argmin(criteria[offset:offset+my_nd]))
                the_list = my_best // my_n
                the_best = my_best % my_n
                the_value = float(values[offset+my_best]) + float(values[offset+my_best+1])
                the_value /= 2
                output[i] = (True, the_list, the_best, float(criteria[offset+my_best])*2/my_n, the_value)

                offset += my_nd
            else:
                offset += d * (abs(positions[i + 1]) - abs(positions[i]))
    
    return output


"""
Fase 3 Realizar división (modificada)
"""
@cuda.jit
def finish_split(attr_list, flag, buffer, false_counter, offset, address):
    idx = cuda.grid(1)
    if idx < buffer.size:
        
        # Si pertenece al split de la dereccha
        # mi posición se encuentra en búfer
        # posición de ínidce saltando los falsos hasta el momento
        if flag[idx]:
            address[idx] = offset[idx] + buffer[idx]
        else:
            # Si no el contador de falsos me indica mi posición
            # (split izquierdo)
            address[idx] = offset[idx] + false_counter[idx] - 1
            
            
        # Realizamos los cambios en la lista de atributos
        # [Valores]
        buffer[idx] = attr_list[0, idx]
        cuda.syncthreads()
        attr_list[0, address[idx]] = buffer[idx]
        cuda.syncthreads()
        # [Clase]
        buffer[idx] = attr_list[1, idx]
        cuda.syncthreads()
        attr_list[1, address[idx]] = buffer[idx]
        cuda.syncthreads()
        # [ID Muestra]
        buffer[idx] = attr_list[2, idx]
        cuda.syncthreads()
        attr_list[2, address[idx]] = buffer[idx]
        cuda.syncthreads()
            


def make_my_splits(attr_lists, positions, phase2, tpb=128):
    # 3.1 Creamos el best_flag para todos los atributos.
    #     Aprovechando este procedimiento actualizamos posiciones.
    best_flag = cp.ones(attr_lists[0][0].size, dtype=cp.bool)
    offset = cp.zeros(attr_lists[0][0].size, dtype=cp.int32)
    address = cp.empty(attr_lists[0][0].size, dtype=cp.int32)
    buffer = cp.empty(attr_lists[0][0].size, dtype=cp.float32)
    false_counter = cp.empty(attr_lists[0][0].size, dtype=cp.int32)
    
    new_positions = np.array([0])
    
    #inicio = time.time()
    # Recorremos los nodos viendo si están activos
    for i in range(len(positions)-1):
        # Si está activo
        if positions[i+1] > 0:
            # Tomamos el índice de la lista y el índice de la muestra de la fase 2
            best_list = phase2[i][1]
            best_attr = phase2[i][2] + 1
            left_samples = cp.array(attr_lists[best_list][2,abs(positions[i]):abs(positions[i])+best_attr], dtype=cp.int32)
            best_flag[left_samples] = 0
            # Como está activo añadimos nuestro nuevo punto en el array de posiciones
            new_positions = np.append(new_positions, best_attr + abs(positions[i]))
        else:
            new_positions = np.append(new_positions, positions[i+1])
        # En ambos casos rellenamos el array de offset
        offset[abs(positions[i]):abs(positions[i+1])] = abs(positions[i])
        new_positions = np.append(new_positions, positions[i+1])

    #fin = time.time()
    #print('c', fin-inicio)
        
    #inicio = time.time()
    # Recorremos las listas de atributos
    for j in range(len(attr_lists)):
        my_flag = cp.zeros(attr_lists[0][0].size, dtype=cp.int32)
    
        # Recorremos los nodos viendo si están activos.
        for i in range(len(positions) - 1):
            start = abs(positions[i])
            end = abs(positions[i+1])
            # Si está activo rellenamos flag con el valor correspondiente
            if positions[i+1] > 0:  
                my_flag[start:end] = best_flag[cp.array(attr_lists[j][2,start:end], dtype=cp.int32)]
                   
            if start != end:
                # En caso contrario todos pertenecen a la clase 0, por lo que no hay problema.
                buffer[start:end] = cp.logical_not(my_flag[start:end])
                false_counter[start:end] = cp.cumsum(buffer[start:end])
                total_false = false_counter[end-1]
                buffer[start:end] = cp.arange(0, end-start) -false_counter[start:end] + total_false
             

        # Hacemos el split de la lista de atributos
       
        blocks = buffer.size // tpb + 1
        
        finish_split[blocks, tpb](attr_lists[j], my_flag, buffer, false_counter, offset, address)
        

        
    #fin = time.time()
    #print('d', fin-inicio)  

    return new_positions


def train_decision_tree(my_dataset, max_depth=10, min_samples_per_node=1, tpb=128):
    # 1. Generamos las listas de atributos y las ordenamos
    my_attr_lists = gen_attribute_lists(my_dataset)
    my_attr_lists = cupy_sort_my_attr_lists(my_attr_lists)
    
    # 2. Nodo raíz
    positions = np.array([0, my_attr_lists[0][0].size])
    
    
    # 3. Generamos el árbol
    outputs = []
    for depth in range(max_depth+1):
        if depth < max_depth - 1:
            
            #inicio = time.time()
            output = cupy_find_split_point(my_attr_lists, positions,min_samples_per_node, tpb) 
            #fin = time.time()
            #print('a', fin-inicio)
            #inicio = time.time()
            positions = make_my_splits(my_attr_lists, positions, output)
            #fin = time.time()    
            #print('b', fin-inicio)
        else:
            output = cupy_find_split_point(my_attr_lists, positions,min_samples_per_node, tpb, last=True)
        outputs.append(output)
    
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


def validacion_cruzada(dataset, k=10, depth=5, my_min=1):
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
        model = train_decision_tree(dataset[samples_train], depth, my_min)
        fin = time.time()
        times[i] = fin-inicio
        accuracy[i] = evaluar(dataset[samples_test[i]], model)

        
    return np.mean(accuracy), np.mean(times)


def run_experimento(dataset, depth, x=1):
    np.random.seed(0)
    times = np.zeros(x, np.float32)
    accuracy = np.zeros(x, np.float32)
    for i in range(x):
        #print(f'Iter {i}')
        accuracy[i], times[i] = validacion_cruzada(dataset, k=10, depth=depth, my_min=1)
        
    print('Profundidad', depth, 'Velocidad', np.mean(times), 'Precisión', np.mean(accuracy))

if __name__ == '__main__':
    print('DATASET SPAMBASE')
    dataset = getdataset_spambase()
    for i in range(4,11):
        run_experimento(dataset, i)
    print('DATASET MAGIC04')
    dataset = getdataset_magic()
    for i in range(4,11):
        run_experimento(dataset, i)