import numpy as np
import cupy as cp
import numba
import math
from numba import cuda


"""
Kernels para SOM online
"""

@cuda.jit
def euclidean_distance(ids, samples, weights, out, d):
    """
    Kernel para calcular la distancia euclídea de un vector con 
    el resto de sus pesos.
    :param ids Índice de la muestra considerada.
    :param samples Conjunto de todas las muestras a evaluar.
    :param weights Vector de filas * columnas * d valores con los pesos 
    asociados a cada una de las neuronas.
    :param out Matriz de tamaño filas * columnas para 
    devolver las distancias euclídeas.
    :param d Número de características asociadas a una entrada.
    :pre d en esta implementación está limitado por el tamaño máximo de
    memoria compartida por multiprocesador. En mi caso, los 48 KB que
    tengo para memoria compartida, indican que el valor máximo para 
    p sería de 12000 elementos aproximadamente.
    """
    idx = cuda.grid(1)
    # Fase 1: Ponemos el vector del array en memoria compartida
    shared_vector = cuda.shared.array(shape=0, dtype=numba.float32)
    if cuda.threadIdx.x == 0:
        for i in range(d):
            shared_vector[i] = samples[ids * d + i]
    cuda.syncthreads()

    # Fase 2: Calculamos la distancia euclídea
    if idx * d < weights.size:
        distance = 0
        for i in range(d):
            i_distance = shared_vector[i] - weights[idx*d+i]
            distance += i_distance * i_distance
            
        # Fase 3: Lo escribimos en el array de salida.
        out[idx] = distance

@cuda.jit
def bmu_update(ids, samples, weights, d, bmu_row, 
    bmu_col, cols, eta, sigma_squared):
    """
    Kernel para la actualización de la BMU y su vecindario
    :param ids Índice de la muestra considerada.
    :param samples Conjunto de todas las muestras a evaluar.
    :param weights Vector de M * N * d valores con los pesos 
    asociados a cada una de las neuronas.
    :param d Número de características asociadas a una entrada.
    :param bmu_row Fila en la que se encuentra la Best Matching Unit.
    :param bmu_col Columna en la que se encuentra la Best Matching Unit.
    :param cols Número de columnas de la matriz de pesos.
    :param eta Valor para la tasa de aprendizaje.
    :param sigma_squared Valor de sigma al cuadrado para el
    cáculo del vecindario.
    """
    idx = cuda.grid(1)
    if idx * d < weights.size:
        
        # 1. Medimos la distancia en la matriz del elemento actual a la BMU
        
        current_row = idx // cols
        current_col = idx % cols

        d_f = (current_col - bmu_col) * (current_col - bmu_col)
        d_f += (current_row - bmu_row) * (current_row - bmu_row)
        if d_f <= sigma_squared:
            # 2. Actualizamos acorde a esa distancia y el valor de sigma
            d_f = math.exp(-d_f/(2*sigma_squared))
            for i in range(d):
                weights[idx * d + i] += eta * d_f * (samples[ids * d + i] - weights[idx * d + i])

"""
Kernels para batch SOM
"""

@cuda.jit
def batch_euclidean_distance(samples, weights, distances):
    """
    Kernel para calcular la distancia euclídea de un todas las muestras
    con los pesos de las neuronas.

    :param samples Conjunto de todas las muestras a evaluar.
    :param weights Vector de filas * columnas * d valores con los pesos 
    asociados a cada una de las neuronas.
    :param distances Matriz para guardar las distancias obtenidas.
    """
    idx = cuda.grid(1)
    if idx < distances.size:
        nrows, ncols, d = weights.shape
        nneurons = nrows * ncols
        row = idx // nneurons
        col = idx % nneurons
        wrow = col // ncols
        wcol = col % ncols

        my_distance = 0
        for i in range(d):
            i_distance = samples[row,i] - weights[wrow, wcol, i]
            my_distance += i_distance * i_distance
            
        distances[row, col] = my_distance


@cuda.jit
def prepare_update(bmu_row, bmu_col, samples, num, den, 
    nrows, ncols, sigma_squared):
    """
    Este kernel calcula numerador y denominador de la fórmula para
    la actualización de pesos iterativos del batch mediante
    sumas atómicas.
    :param bmu_row Vector con la fila de la BMU de cada muestra.
    :param bmu_col Vector con la columna de la BMU de cada muestra.
    :param samples Conjunto de las muestras usadas para entrenar.
    :param num Vector con los numeradores para el cálculo de la fórmula.
    :param den Vector con los denominadores para el cálculo de la fórmula.
    :param nrow Número de filas en la capa de salida.
    :param ncols Número de columnas en la capa de salida.
    :param sigma_squared Valor de sigma al cuadrado para el
    cáculo del vecindario.
    """
    idx = cuda.grid(1)
    if idx < bmu_row.size:
        my_row = bmu_row[idx]
        my_col = bmu_col[idx]
        
        init_row = max(0, my_row - int(sigma_squared))
        finish_row = min(nrows, my_row + int(sigma_squared) + 1)
        
        init_col = max(0, my_col - int(sigma_squared))
        finish_col = min(ncols, my_col + int(sigma_squared) + 1)
        
    
        for i in range(init_row, finish_row):
            for j in range(init_col, finish_col):
                dist = (j-my_col) * (j-my_col) + (i-my_row) * (i-my_row)
    
                if dist <= sigma_squared:
                    hck = math.exp(-(dist)/(2 * sigma_squared))
                    cuda.atomic.add(den, i*ncols+j, hck)
                    d = samples.shape[1]
                    for k in range(d):
                        cuda.atomic.add(num, i*ncols*d + j*d +k,
                        hck * samples[idx, k])

        

@cuda.jit
def finish_update(weights, num, den):
    """
    Este kernel actualiza los pesos para la siguiente iteración
    con los datos del numerador y el denominador.
    :param weights Vector de filas * columnas * d valores con los pesos 
    asociados a cada una de las neuronas.
    :param num Vector con los numeradores para el cálculo de la fórmula.
    :param den Vector con los denominadores para el cálculo de la fórmula.
    """
    idx = cuda.grid(1)
    if idx < den.size:
        nrows, ncols, d = weights.shape
        row = idx // ncols
        col = idx % ncols
        my_den = den[row * ncols + col]
        if my_den != 0:
            for k in range(d):
                weights[row, col, k] = num[row*ncols*d + col*d +k] / my_den
      
        cuda.syncthreads()
       
        for k in range(d):
            num[row*ncols*d + col*d +k] = 0
            
        den[row * ncols + col] = 0