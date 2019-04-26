import numpy as np
from numba import cuda, float32, int32, int64, void, jit
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import time
import math
import pyculib
"""
Implementación de mapas autoorganizados o redes de Kohonen en CUDA con Numba.
Se considera una versión online del algoritmo (analiza las muestras de una en una, no en batchs).

En esta implementación suponemos que trabajamos con una red bidimensional (fils x cols)
de neuronas con d pesos sinápticos.
"""

"""
Para la implementación de la versión tradicional del algoritmo hemos necesido utilizar 4 kernels.
Debido a la secuencialidad intrínseca del algoritmo para procesar las muestras el grado de paralelismo
depende del número de neuronas en la capa de salida.

El primer kernel se encarga de inicializar aleatoriamente la matriz de los pesos asociadas a las neuronas.
Para realizar esta generación aleatoria en la GPU utilizamos las herramientas que numba cuda nos proporciona.

El segundo kernel se encarga de calcular la distancia euclídea de la muestra con todos los pesos de la capa
de salida.

El tercer kernel realiza una reducción para encontrar la unidad con menor distancia a la muestra proporcionada.
Mientras que hemos propuesto una implementación de cómo se haría, hemos optado por utilizar la función
amin() de cuBLAS que nos proporciona resultados más rápido.

El último kernel actualiza la matriz de pesos.
"""

"""
Kernel para inicializar aleatoriamente la 'matriz' de pesos con valores 
en el intervalo [0, 1) tomados de una distribución aleatoria
:param rng_states Estados aleatorios
:param weigths Vector de M * N * d valores que contendrá los pesos asociados a las neuronas
"""
@cuda.jit
def cuda_init_weights(rng_states, weights):
    idx = cuda.grid(1)
    if idx < weights.size:
        weights[idx] = xoroshiro128p_uniform_float32(rng_states, idx)


"""
Kernel para calcular la distancia euclídea de un vector con el resto de sus pesos.
:param ids Índice de la muestra considerada.
:param samples Vector con todas las muestras.
:param weights Vector de M * N * p valores con los pesos asociados a cada una de las neuronas.
:param out Vector de tamaño M * N para devolver las distancias euclídeas.
:param d Número de características asociadas a una entrada.
:pre d en esta implementación está limitado por el tamaño máximo de memoria compartida por
     multiprocesador. En mi caso, los 48 KB que tengo para memoria compartida, indican que 
     el valor máximo para p sería de 12000 elementos aproximadamente.
     
"""

@cuda.jit('(uint64, float32[:], float32[:], float32[:], int32)', fastmath=True)
def cuda_euclidean_distance(ids, samples, weights, out, d):
    idx = cuda.grid(1)
    # Fase 1: Ponemos el vector del array en memoria compartida
    shared_vector = cuda.shared.array(shape=0, dtype=float32)
    if idx < d:
        shared_vector[idx] = samples[ids * d + idx]
    cuda.syncthreads()

    # Fase 2: Calculamos la distancia euclídea
    if idx * d < weights.size:
        distance = 0
        for i in range(d):
            i_distance = shared_vector[i] - weights[idx*d+i]
            distance += i_distance * i_distance
            
        # Fase 3: Lo escribimos en el array de salida.
        out[idx] = math.sqrt(distance)

"""
El siguiente kernel realizado encuentra el valor con la menor distancia posible.
Este kernel ha sido implementado mediante la siguiente reducción.
Puesto que el kernel creado es entre 2 y 4 veces más lento que la implementación
de amin de cuBLAS se propone utilizar dicha opción para esa operación.

El problema de la implementación realizada es hemos de llamar múltiples veces al mismo
kernel mientras que probablemente la versión de cuBLAS utilice alguna versión utilizando atomics
u optimizaciones más avanzadas que la versión propuesta.
"""


@cuda.jit('(float32[:], float32[:], int32[:])')
def cuda_min_element(my_array, my_mins, my_indexes):
    # 1. Declaramos la memoria compartida
    shared_mem = cuda.shared.array(shape=128, dtype=float32)
    shared_idx = cuda.shared.array(shape=128, dtype=int32)
    
    # 2. Obtenemos los índices
    tidx = cuda.threadIdx.x
    idx = cuda.blockDim.x * cuda.blockIdx.x + tidx
    
    # 3. Inicialiamos a infinito
    shared_mem[tidx] = np.inf
    
    # 4. Cada thread comprueba un stride del grid
    while idx < my_array.size:
        if my_array[idx] < shared_mem[tidx]:
            shared_mem[tidx] = my_array[idx]
            shared_idx[tidx] = idx
                      
        idx += cuda.blockDim.x * 2 * cuda.gridDim.x
    cuda.syncthreads()
    
    idx = cuda.blockDim.x * cuda.blockIdx.x + tidx
    s = cuda.blockDim.x // 2
    
    # 5. Realizamos el proceso de reducción
    while s > 32:
        if tidx < s and idx < my_array.size:
            if shared_mem[tidx + s] < shared_mem[tidx]:
                shared_mem[tidx] = shared_mem[tidx + s]
                shared_idx[tidx] = shared_idx[tidx + s]
        cuda.syncthreads()
        s //= 2
    
    # 6. Hacemos unroll para un warp (nos ahorramos syncthreads)
    if tidx < 32:
        if shared_mem[tidx + 32] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 32]
            shared_idx[tidx] = shared_idx[tidx + 32]
        if shared_mem[tidx + 16] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 16]
            shared_idx[tidx] = shared_idx[tidx + 16]
        if shared_mem[tidx + 8] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 8]
            shared_idx[tidx] = shared_idx[tidx + 8]
        if shared_mem[tidx + 4] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 4]
            shared_idx[tidx] = shared_idx[tidx + 4]
        if shared_mem[tidx + 2] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 2]
            shared_idx[tidx] = shared_idx[tidx + 2]
        if shared_mem[tidx + 1] < shared_mem[tidx]:
            shared_mem[tidx] = shared_mem[tidx + 1]
            shared_idx[tidx] = shared_idx[tidx + 1]
            
            
    # El primer thread de cada bloque indica su minimo
    # Si da para más de un bloque, luego hay que reaplicar el kernel
    if tidx == 0:
        my_mins[cuda.blockIdx.x] = shared_mem[tidx]
        my_indexes[cuda.blockIdx.x] = shared_idx[tidx]

    

def reduce_min_index(a):
    list_indexes = []
   
    
    def set_kernel_params(a):
        threads = min(a.size, 128)
        blocks = a.size // threads
        sm_size = threads * 2 * 4
        mins = np.empty(blocks, np.float32)
        indexes = np.empty(blocks, np.int32)
        return blocks, threads, 0, sm_size, a, mins, indexes
        
    def run_kernel(p):
        cuda_min_element[p[0], p[1], p[2], p[3]](p[4], p[5], p[6])
        
    p = set_kernel_params(a)
    if p[0] == 1:
        run_kernel(p)
        return p[6][0]
    else:
        while p[0] > 1:
            run_kernel(p)
            list_indexes.append(p[6])
            a = p[5]
            p = set_kernel_params(a)
        run_kernel(p)
        list_indexes.append(p[6])
        
        output_index = 0
        for indexes in list_indexes[::-1]:
            output_index = indexes[output_index]
            
        return output_index
            
    
"""
El tercer kernel se encarga de actualizar la matriz de pesos
:param ids Índice de la muestra considerada.
:param samples Vector con todas las muestras.
:param weights Vector de M * N * p valores con los pesos asociados a cada una de las neuronas.
:param d Número de características asociadas a una entrada.
:param bmu_row Fila en la que se encuentra la Best Matching Unit.
:param bmu_col Columna en la que se encuentra la Best Matchin Unit.
:param cols Número de columnas de la matriz de pesos.
:param eta Valor para la tasa de aprendizaje.
:param sigma_squared Valor de sigma al cuadrado para el cáculo del vecindario.
"""

@cuda.jit('uint64, float32[:],  float32[:], uint64, uint64, uint64, uint64, float32, float32', fastmath=True)
def cuda_bmu_update(ids, samples, weights, d, bmu_row, bmu_col, cols, eta, sigma_squared):
    idx = cuda.grid(1)
    if idx * d < weights.size:
        
        # 1. Medimos la distancia en la matriz del elemento actual a la BMU
        
        current_row = idx // cols
        current_col = idx % cols

        d_f = (current_col - bmu_col) * (current_col - bmu_col)
        d_f += (current_row - bmu_row) * (current_row - bmu_row)
        if d_f < sigma_squared:
            # 2. Actualizamos acorde a esa distancia y el valor de sigma
            d_f = math.exp(- float(d_f)/2/sigma_squared)
            for i in range(d):
                weights[idx * d + i] += eta * d_f * (samples[ids * d + i] - weights[idx * d + i])
            
"""
Con todos los kernels creamos una función para encapsular todo el procedimiento del algoritmo.
Vamos a manejar todos la memoria de manera manual para asegurarnos de que no se hace ninguna 
transferencia innecesaria de memoria entre host y dispositivo.

En esta implementación se supone que queremos obtener la capa de salida de la red en una estructura bidimensional 'matriz'.
:param samples Muestras con las que entrenar el modelo.
:param rows Número de filas de la matriz de salida.
:param cols Número de columnas de la matriz de salida.
:param iters Número de iteraciones máximas.
:param nsamples Número de muestras a considerar por iteración.
:param sigma_0 Valor de sigma inicial para el cálculo del vecinadrio.
:param sigma_f Valor de sigma una vez ha sido fijado para el cálculo del vecinadrio.
:param eta_0 Valor de eta inicial para la tasa de aprendizaje.
:param eta_f Valor de eta una vez ha sido fijado para la tasa de aprendizaje.
:param tau Constante para la gaussiana decreciente
:param iter_smooth Número de iteración en la que los valores de eta y sigma pasan a ser constantes.
:param tpb Número de hebras por bloque.
:param seed Semilla para reproducción de resultados.
:return Matriz con los pesos finales
"""
def sofm(samples, rows, cols, iters, nsamples, sigma_0, sigma_f, eta_0, eta_f, tau, iter_smooth, tpb=128, seed=None):
    np.random.seed(seed)
    # 0. Preparamos la memoria del dispositivo
    d = samples[0].size
    d_weights = cuda.device_array(rows * cols * d, np.float32)
    d_samples = cuda.to_device(samples.flatten())
    distances = cuda.device_array(rows * cols, np.float32)
    
    # 1. Inicializamos la matriz de pesos.
    # 1.a Preparamos parámetros para kernel.
    
    blocks = rows * cols * d // tpb + 1
    # 1.b Generamos estados aleatorios.
    rng_states = create_xoroshiro128p_states(blocks * tpb, seed=seed)
    # 1.c Invocamos el kernel.
    cuda_init_weights(rng_states, d_weights)
    
      
    # 2. Bucle principal de algoritmo
    # 2.a Preparamos los parámetros para el resto de kernels
    sm_size = samples[0].size * samples[0].dtype.itemsize
    blocks = rows * cols // tpb + 1
    cuBLAS = pyculib.blas.Blas()
    
    samples_indexes = np.arange(len(samples))
    # En cada iteración de entrenamiento del algoritmo
    for t in range(iters):
        # 2.b Actualizamos los parámetros que depende de la iteración
        if t < iter_smooth:
            sigma = float(sigma_0) * np.exp(float(-t)/tau)
            eta = float(eta_0) * np.exp(float(-t)/tau)
        else:
            sigma = sigma_f
            eta = eta_f
        selected_samples = np.random.choice(samples_indexes, nsamples)
        # Para cada muestra
        for i in selected_samples:
            # 2.c Calculamos la distancia euclídea
            cuda_euclidean_distance[blocks, tpb, 0, sm_size](i, d_samples, d_weights, distances, d)
            # 2.d Obtenemos el índice de la menor distancia
            min_index = cuBLAS.amin(distances)
            # 2.e Actualizamos la matriz de los pesos
            bmu_row = min_index // cols
            bmu_col = min_index % cols
            cuda_bmu_update[blocks, tpb](i, d_samples, d_weights, d, bmu_row, bmu_col, cols, eta, sigma * sigma)

        
    # 3. Devolver el resultado
    return d_weights.copy_to_host()
            
            
if __name__ == '__main__':
	"""
    Realicemos un ejemplo para comprobar el correcto funcionamiento del algoritmo.
    """
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces()

    inicio = time.time()
    weights = sofm(faces['data'], 20, 20, 400, 100, 10, 0.01, 1, 0.1, 400, 100, 128, seed=20)
    fin = time.time()
    print(f'El algoritmo ha tardado {fin-inicio} segundos en ejecutarse.')

    W = weights.reshape((20, 20, faces['data'].shape[1]))
    pattern_length = faces['data'].shape[1]
    pattern_width = pattern_height = int(np.sqrt(pattern_length))
    matrix_side = 20
    matrix_w = np.empty((matrix_side * pattern_height, matrix_side * pattern_width))
    matrix_w *= 255


    for i in range(matrix_side):
        for j in range(matrix_side):
            matrix_w[i * pattern_height:i * pattern_height + pattern_height, 
    j * pattern_height:j * pattern_height + pattern_width] = W[i, j].reshape((pattern_height, pattern_width)) * 255.0
    fig, ax = plt.subplots(figsize=(12,12))

    ax.matshow(matrix_w.tolist(), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
      
    plt.show()
        
        
