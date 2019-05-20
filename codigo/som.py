from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import numba
import math
import utils
import time
"""
Kernels Numba para batch SOM
Necesitamos:
a) Calcular las distancias euclídas entre un
   conjunto de muestras y los pesos de las neuronas.
b) Encontrar el índice de los pesos de neuronas con distancia mínima.
c) Actualizar los pesos de las neuronas conforme a la ecuación correspondiente.
"""

"""
A) Cálculo de distancia euclídea.
Puesto que lo usamos para la relación de orden podemos obviar el cálculo de la raíz cuadrada.

Uso de memoria compartida y bloques
------------------------------------
Mantenemos los pesos de la neurona del bloque en memoria compartida y 
calculamos la distancia de la neurona con el conjunto de muestras.

Si
N= Nº de muestras
nneurons = filas * columnas = Nº neuronas
d = dimensión del problema
tpb = Thhreads per Block

necesitamos lanzar un grid de (nneurons, N // tpb + 1) bloques de tpb hebras
"""

@cuda.jit
def euclidean_distance(samples, weights, distances, nsamples, d):
    """
    Kernel para calcular la distancia euclídea de todas las muestras
    con los pesos
    :param samples Conjunto de todas las muestras a evaluar.
    :param weights Array de filas * columnas * d valores con los pesos 
    asociados a cada una de las neuronas.
    :param distances Array de tamaño N * nneurons para almacenar
    las distancias
    :param nsamples Número total de muestras.
    :param d Dimensión de cada muestra.
    """
    # 1. Tomamos los índices que nos correspoden
    neuron_idx = cuda.blockIdx.x
    samples_idx = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    
    
    # 2. Ponemos los pesos de la neurona en memoria compartida
    shared_weights = cuda.shared.array(shape=0, dtype=numba.float32)
    for i in range(d // cuda.blockDim.x + 1):
        i_stride = i * cuda.blockDim.x
        my_pos = i_stride + cuda.threadIdx.x
        if my_pos < d:
            shared_weights[my_pos] = weights[neuron_idx * d + my_pos]
            
    cuda.syncthreads()
    
    # 3. Procedemos a realizar el cálculo de la distancia si procede
    if samples_idx < nsamples:
        distance = 0.0
        for i in range(d):
            i_distance = samples[samples_idx * d + i] - shared_weights[i]
            distance += i_distance * i_distance
            
        distances[samples_idx, neuron_idx] = distance


"""
C) Actualización de los pesos de las neuronas.
La fórmula se corresponde a un cociente de sumatorias.
Para resolverlo, en el kernel prepare_update se realizan
los cálculos del numerador y del denominador por separado.

Uso de memoria compartida y bloques
------------------------------------
En este caso, usamos la memoria compartida para asegurarnos de que
la muestra está en caché y no tenga que ser cargada de nuevo mientras
se evalúan todas las neuronas.

necesitamos lanzar un grid de (N, nneurons // tpb + 1) bloques de tpb hebras
"""

@cuda.jit
def prepare_update(bmu, samples, num, den, 
    nrows, ncols, d, sigma_squared):
    """
    Este kernel calcula numerador y denominador de la fórmula para
    la actualización de pesos iterativos del batch mediante
    sumas atómicas.
    :param bmu Vector con las posiciones de la BMU.
    :param samples Conjunto de las muestras usadas para entrenar.
    :param num Vector con los numeradores para el cálculo de la fórmula.
    :param den Vector con los denominadores para el cálculo de la fórmula.
    :param nrows Número de filas en la capa de salida.
    :param ncols Número de columnas en la capa de salida.
    :param d Dimensión de cada muestra.
    :param sigma_squared Valor de sigma al cuadrado para el
    cáculo del vecindario.
    """
    # 1. Tomamos los índices que correspondan
    sample_idx = cuda.blockIdx.x
    neuron_idx = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
   
    # 2. Metemos en memoria compartida la muestra que se lee en todo el bloque
    shared_sample = cuda.shared.array(shape=0, dtype=numba.float32)
    for i in range(d // cuda.blockDim.x + 1):
        i_stride = i * cuda.blockDim.x
        my_pos = i_stride + cuda.threadIdx.x
        if my_pos < d:
            shared_sample[my_pos] = samples[sample_idx * d + my_pos]
    cuda.syncthreads()
    
    # 3. Si procede realizar cálculos los hacemos
    if neuron_idx < nrows * ncols:
        bmu_row = bmu[sample_idx] // ncols
        bmu_col = bmu[sample_idx] % ncols
        neuron_row = neuron_idx // ncols
        neuron_col = neuron_idx % ncols
        
        dist = (neuron_row - bmu_row) * (neuron_row - bmu_row) + (neuron_col - bmu_col) * (neuron_col - bmu_col)
        
        if dist <= sigma_squared:
            hck = math.exp(-dist/(2 * sigma_squared))
            # Guardamos sumatoria del denominador
            cuda.atomic.add(den, neuron_row * ncols + neuron_col, hck)
            # Guardamos sumatoria del numerador
            for i in range(d):
                cuda.atomic.add(num, neuron_row * ncols * d + neuron_col * d + i, hck * shared_sample[i])

@cuda.jit
def finish_update(weights, partials, numParts, nrows, ncols, d):
    """
    Este kernel terminas las sumas parciales.
    Se ejecuta en un único nodo de Spark.
    
    :param weights Array de pesos de neuronas
    :param partials Array con sumas parciales
    :param numParts Número de resultados parciales a procesar.
    :param nrows Número de filas en la capa de salida.
    :param ncols Número de columnas en la capa de salida.
    :param d Dimensión de cada muestra.
    
    Estrucutra de bloques
    ---------------------
    Lanzamos nrows * ncols // tpb + 1 bloques.
    """
    idx = cuda.grid(1)
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
                partials[row * ncols * d + col * d + k] += partials[fullsize * i + row * ncols * d + col * d + k]
            # Suma de denominadores
            partials[numsize + row * ncols + col] += partials[fullsize * i + numsize + row * ncols + col]
            
    cuda.syncthreads()
    
    if idx < nrows * ncols:
        # b) Si no es 0 el denominador realizamos la división y cambiamos pesos actuales
        if partials[numsize + row * ncols + col] != 0:
            for k in range(d):
                weights[row * ncols * d + col * d + k] =  partials[row*ncols*d + col*d +k] / \
                    partials[numsize + row * ncols + col]
      
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
Spark con GPU
"""
def gpu_work_iter(d, rows, cols, weights, sigma_squared, tpb=128):
    # Declarada función interna para devolverla y poder utilizar
    # múltiples parámetros al llamar a mapPartitions
    def _gpu_work(data):
        # 1. Procesamos el dataset
        inp = np.asarray(list(data), dtype=np.float32)
        N = inp.size // d
       
        nneurons = rows * cols
        dsamples = cuda.to_device(inp.ravel())
        dweights = cuda.to_device(weights)
        # 2. Generamos un estructura en el dispositivo para almacenar las distancias
        distances = cuda.device_array((N, nneurons), dtype=np.float32)
        
        # 3. Calcuamos las distancias
        sm_size = 4 * d
        euclidean_distance[(nneurons, N // tpb + 1),tpb, 0, sm_size](dsamples, dweights, distances, N, d)
        
        # 4. Realizamos las reducciones para todas las muestras
        bmu = utils.multi_reduce_min_index(distances)

        # 5. Calculamos el numerador y denominador parcial asociado
        num = np.zeros(rows * cols * d, np.float32)
        den = np.zeros(rows * cols, np.float32)
        
        dnum = cuda.to_device(num)
        dden = cuda.to_device(den)
        prepare_update[(N, (rows * cols) // tpb + 1), tpb,0, sm_size](bmu, dsamples, dnum, dden, rows, cols, d, sigma_squared)
        
        return dnum.copy_to_host(), dden.copy_to_host()
    return _gpu_work

def spark_gpu_batch_som(rdd_data, d, max_iters, rows, cols, smooth_iters=None, sigma_0=10, 
                          sigma_f=0.1, tau=400, seed=None, tpb=128):
    
    # 1. Inicializamos pesos aleatorios
    blocks = rows * cols * d // tpb + 1
    d_weights = cuda.device_array(rows * cols * d, np.float32)
    
    rng_states = create_xoroshiro128p_states(rows * cols * d, seed=seed)
    cuda_init_weights[rows * cols * d // tpb + 1, tpb](rng_states, d_weights)
    
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
        out = rdd_data.mapPartitions(gpu_work_iter(d, rows, cols, weights, sigma_squared))
        
        # 2.c En un único nodo usamos la GPU para juntar todas las sumas parciales obtenidas
        #   y realizar la división
        out = out.collect()
        numParts = len(out) // 2

        partials = np.concatenate(out)
        finish_update[rows * cols // tpb + 1, tpb](weights, partials, numParts, rows, cols, d)
       
    return weights



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