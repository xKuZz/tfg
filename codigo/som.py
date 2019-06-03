from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import numba
import math
import time


@cuda.jit
def rand_weights(rng_states, d_weights):
    """
    Kernel para inicializar aleatoriamente la 'matriz' de pesos con valores 
    en el intervalo [0, 1) tomados de una distribución aleatoria
    :param rng_states Estados aleatorios
    :param d_weigths Vector de M * N * d valores que contendrá los pesos asociados a las neuronas
    """
    idx = cuda.grid(1)
    # Cogemos índices para pasar de array unidimensional a tridimensional
    n_rows, n_cols, d = d_weights.shape
    row = idx // (n_cols * d)
    col_d = idx % (n_cols * d)
    col = col_d // d
    i = col_d % d
    
    # Sacamos el aleatorio correspondiente
    if idx < d_weights.size:
        d_weights[row, col, i] = xoroshiro128p_uniform_float32(rng_states, idx)

@cuda.jit
def som_iter(d_samples, d_weights, d_nums, d_denums, sigma_squared):
    """
    Este kernel realiza el proceso de calcular las distancias euclídeas entre
    todas las muestras y los pesos de las neuronas. Encontrar la mejor BMU para
    una muestra y realizar el cálculo parcial de los pesos correspondientes.
    :param d_samples Conjunto de todas las muestras a evaluar.
    :param d_weights Array de filas * columnas * d valores con los pesos 
    asociados a cada una de las neuronas.
    :param d_distances Array de tamaño N * nneurons para almacenar
    las distancias
    :param d_nums Vector con los numeradores para el cálculo de la fórmula.
    :param d_denums Vector con los denominadores para el cálculo de la fórmula.
    :param sigma_squared Valor de sigma al cuadrado para el cáculo del vecindario.
    """
    # 0. Índices
    nrows, ncols, d = d_weights.shape
    nneurons = nrows * ncols
    
    
    sample_idx = cuda.blockIdx.x
    neuron_idx = cuda.threadIdx.x
    neuron_row = neuron_idx // ncols
    neuron_col = neuron_idx % ncols
    blockSize = cuda.blockDim.x
       
    # 0. Declaramos  e inicializamos la memoria compartida
    # Memoria compartida para guardar la muestra del bloque
    shared_sample = cuda.shared.array(shape=0, dtype=numba.float32)
    # Memoria compartida para guardar las distancias de cada muestra del bloque con
    # cada neurona (máximo 1024 neuronas).
    shared_distances = cuda.shared.array(shape=1024, dtype=numba.float32)
    # Memoria compartida para los índices de la reducción (máximo 1024 neuronas).
    shared_idx = cuda.shared.array(shape=1024, dtype=numba.int32)
    
    # 1. Empezamos calculando la distancia euclídea de la muestra con las neuronas
    #    del bloque.
    # 1.a Cargamos la muestra del bloque en memoria compartida 
    for i in range(d // nneurons + 1):
        i_stride = i * nneurons
        my_pos = i_stride + cuda.threadIdx.x
        if my_pos < d:
            shared_sample[my_pos] = d_samples[sample_idx, my_pos]
    
    
    cuda.syncthreads()
    
    # 1.b Calculamos las distancias euclídeas que nos corresponden.
    # Aprovechamos la barrera al final de la operación para inicializar la
    # memoria compartida para la reducción.
    
    if neuron_idx < nneurons:
        shared_distances[neuron_idx] = 0.0
        for i in range(d):
            i_distance = shared_sample[i] - d_weights[neuron_row, neuron_col, i]
            shared_distances[neuron_idx] += i_distance * i_distance
    else:
        shared_distances[neuron_idx] = np.inf
    
    # Inicializamos el array de índices para la reducción.
    shared_idx[neuron_idx] = neuron_idx
    cuda.syncthreads()
    
    
    # 2. Realizamos la reducción para encontrar la mejor distancia.
    
    # Unroll de bloque
    if blockSize >= 1024 and neuron_idx < 512:
        if shared_distances[neuron_idx + 512] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 512]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 512]
    cuda.syncthreads()
    
    if blockSize >= 512 and neuron_idx < 256:
        if shared_distances[neuron_idx + 256] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 256]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 256]
    cuda.syncthreads()
    
    if blockSize >= 256 and neuron_idx < 128:
        if shared_distances[neuron_idx + 128] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 128]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 128]
    cuda.syncthreads()
    
    if blockSize >= 128 and neuron_idx < 64:
        if shared_distances[neuron_idx + 64] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 64]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 64]
    cuda.syncthreads()
    
    # Unroll de warp
    if neuron_idx < 32:
        if shared_distances[neuron_idx + 32] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 32]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 32]
        if shared_distances[neuron_idx + 16] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 16]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 16]
        if shared_distances[neuron_idx + 8] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 8]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 8]
        if shared_distances[neuron_idx + 4] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 4]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 4]
        if shared_distances[neuron_idx + 2] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 2]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 2]
        if shared_distances[neuron_idx + 1] < shared_distances[neuron_idx]:
            shared_distances[neuron_idx] = shared_distances[neuron_idx + 1]
            shared_idx[neuron_idx] = shared_idx[neuron_idx + 1]
    
    cuda.syncthreads()
    
    # La mejor distancia se encuentra en la posición 0 del array.
    bmu = shared_idx[0]
    bmu_row = bmu // ncols
    bmu_col = bmu % ncols

    cuda.syncthreads()
    # 3. Realizamos la actualización de los pesos.
    if neuron_idx < nneurons:
        dist = (neuron_row - bmu_row) * (neuron_row - bmu_row) + \
               (neuron_col - bmu_col) * (neuron_col - bmu_col)
        # Si estamos dentro del rango de actualización.
        if dist <= sigma_squared:
            hck = math.exp(-dist/(2 * sigma_squared))
            # Guardamos sumatoria del denominador
            cuda.atomic.add(d_denums, neuron_row * ncols + neuron_col, hck)
            # Guardamos sumatoria del numerador
            for i in range(d):
                cuda.atomic.add(d_nums, neuron_row*ncols*d + neuron_col*d+i,
                                hck * shared_sample[i])


@cuda.jit
def finish_update(d_weights, partials, numParts):
    """
    Este kernel terminas las sumas parciales.
    Se ejecuta en un único nodo de Spark.
    
    :param d_weights Array de pesos de neuronas
    :param partials Array con sumas parciales
    :param numParts Número de resultados parciales a procesar.
    """
    idx = cuda.grid(1)
    nrows, ncols, d = d_weights.shape
    if idx < nrows * ncols:
        row = idx // ncols
        col = idx % ncols
        
        # a) Sumamos todos los parciales en el primer array
        numsize = nrows * ncols * d
        densize = nrows * ncols
        fullsize = numsize + densize
        for i in range(numParts - 1):
            # Suma de numeradores
            for k in range(d):
                pos = fullsize * i + row * ncols * d + col * d + k
                partials[row * ncols * d + col * d + k] += partials[pos]
            # Suma de denominadores
            pos = fullsize * i + numsize + row * ncols + col
            partials[numsize + row * ncols + col] += partials[pos]
    
        # b) Si no es 0 el denominador realizamos la división y cambiamos pesos actuales
        if partials[numsize + row * ncols + col] != 0:
            for k in range(d):
                d_weights[row, col, k] = partials[row*ncols*d + col*d +k] / \
                                         partials[numsize + row * ncols + col]


def gpu_work_iter(weights, sigma_squared):
    # Declarada función interna para devolverla y poder utilizar
    # múltiples parámetros al llamar a mapPartitions
    def _gpu_work(data):
        # 1. Procesamos el dataset
        inp = np.asarray(list(data), dtype=np.float32)
        rows, cols, d = weights.shape
        nneurons = rows * cols
        N = inp.shape[0]
    
        # 2. Pasamos los datos a las memorias del dispositivo
        d_samples = cuda.to_device(inp)
        d_weights = cuda.to_device(weights)
        nums = np.zeros(rows * cols * d, np.float32)
        denums = np.zeros(rows * cols, np.float32)
        d_nums = cuda.to_device(nums)
        d_denums = cuda.to_device(denums)
        
        # 3. Tomamos el número de hebras por bloque
        if nneurons > 1024:
            raise Exception('Número de neuronas superior al límite')
        
        tpb = max(64,2**(math.ceil(math.log2(nneurons))))
            
        # 3. Lanzamos el kernel.
        sm_size = 4 * d # Memoria compartida para almacenar una muestra por bloque
        som_iter[N, tpb, 0, sm_size](d_samples, d_weights, d_nums, d_denums, sigma_squared)
        
        return d_nums.copy_to_host(), d_denums.copy_to_host()
    return _gpu_work

def spark_gpu_batch_som(rdd_data, d, max_iters, rows, cols, smooth_iters=None, sigma_0=10, 
                          sigma_f=0.1, tau=400, seed=None, tpb=1024):
    
    # 1. Inicializamos pesos aleatorios
    d_weights = cuda.device_array((rows, cols ,d), np.float32)
    rng_states = create_xoroshiro128p_states(rows * cols * d, seed=seed)
    rand_weights[(d_weights.size) // tpb + 1, tpb](rng_states, d_weights)
     
    weights = d_weights.copy_to_host()
    # 2. Bucle del algoritmo
    for t in range(max_iters):
        # 2.a Actualizamos los parámetros de control si procede
        if smooth_iters is None or t < max_iters:
            sigma = sigma_0 * math.exp((-t/tau))
        else:
            sigma = sigma_f
            
        sigma_squared = sigma * sigma
        
        # 2.b Cada nodo del clúster de spark trabajará con un subconjunto
        # de las muestras del RDD para encontrar la BMU y realizar la suma
        # parcial de su ecucación de actualización de pesos
        out = rdd_data.mapPartitions(gpu_work_iter(weights, sigma_squared))
        
        # 2.c En un único nodo usamos la GPU para juntar todas las sumas parciales obtenidas
        #   y realizar la división
        out = out.collect()
        numParts = len(out) // 2

        partials = np.concatenate(out)
        finish_update[rows * cols // tpb + 1, tpb](weights, partials, numParts)
       
    return weights

            
"""
Medidas de calidad
"""
def quantification_error(dataset, weights):
    error = 0.0
    for sample in dataset:
        error += np.min(np.linalg.norm(sample - weights,axis=2))

    return error / len(dataset)

def topography_error(dataset, weights):
    error = 0
    for sample in dataset:
        distances = np.linalg.norm(sample - weights, axis=2)

        bmus = np.argsort(distances, axis=None)[:2]
        bmu_rows = bmus // weights.shape[1]
        bmu_cols = bmus % weights.shape[1]

        if abs(bmu_rows[1] - bmu_rows[0]) == 1 and abs(bmu_cols[1] - bmu_cols[0]) == 1:
            error += 1

    return error / len(dataset)

"""
Spark con CPU
"""
def cpu_work_iter(d, rows, cols, weights, sigma_squared, tpb=128):
    # Declarada función interna para devolverla y poder utilizar
    # múltiples parámetros al llamar a mapPartitions
    def _cpu_work(data):
        # 1. Procesamos el dataset
        inp = np.asarray(list(data), dtype=np.float32)
        N = inp.size // d
        inp = np.resize(inp,(N, d))
        nneurons = rows * cols
        # 2. Calculamos las distancias
       
        bmus = np.array([np.argmin(np.linalg.norm(sample - weights, axis=2)) for sample in inp])
        bmu_rows, bmu_cols = bmus // cols, bmus % cols

        nums = np.zeros(weights.shape, np.float32)
        dens = np.zeros((rows, cols), np.float32)

        # 2.3 Generamos las sumas parciales de los nuevos pesos
        for x, (row, col) in enumerate(zip(bmu_rows, bmu_cols)):
            for i in range(rows):
                for j in range(cols):
                    hck = (j - col) * (j - col) + (i - row) * (i - row)
            
                    if hck <= sigma_squared:
                        hck = math.exp(-hck/(2 * sigma_squared))
                        dens[i, j] += hck
                        nums[i, j, :] += hck * inp[x]
        
        return nums, dens
    return _cpu_work

def spark_cpu_batch_som(rdd_data, d, max_iters, rows, cols, smooth_iters=None, sigma_0=10, 
                          sigma_f=0.1, tau=400, seed=None):
    
    # 1. Inicializamos pesos aleatorios
    weights = np.random.ranf((rows, cols, d))
         
    # 2. Bucle del algoritmo
    for t in range(max_iters):
        # 3.a Actualizamos los parámetros de control si procede
        if smooth_iters is None or t < max_iters:
            sigma = sigma_0 * math.exp((-t/tau))
        else:
            sigma = sigma_f
            
        sigma_squared = sigma * sigma
        
        # 2.b Cada nodo del clúster de spark trabajará con un subconjunto
        # de las muestras del RDD para encontrar la BMU y realizar la suma
        # parcial de su ecucación de actualización de pesos
        out = rdd_data.mapPartitions(cpu_work_iter(d, rows, cols, weights, sigma_squared))
        
        # 2.c Recogemos los datos obtenidos y realizamos la división para actualizar
        out = out.collect()
        numParts = len(out) // 2

        nums = np.zeros(weights.shape, dtype=np.float32)
        dens = np.zeros((rows, cols), dtype=np.float32)

        for i in range(numParts):
            nums += out[2*i]
            dens += out[2*i+1]

        for i in range(rows):
                for j in range(cols):
                    if dens[i, j] != 0:
                        weights[i, j] = nums[i, j] / dens[i, j]

       
    return weights