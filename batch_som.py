#!/usr/bin/env python
# coding: utf-8

"""
TRABAJO EN PROGRESO!!!
Código operativo, estudiando si puede haber alguna mejora en vez de utilizar atomics o cambiar por CPU ese fragmento.

El SOM batch a diferencia del som tradicional (on-line) procesa el conjunto de muestras a la vez durante una
serie de iteraciones. A cada una de esta iteraciones, se le denomina época y, el vector de pesos asociados a las neuronas,
se actualiza tomando la media de las muestras que han activado dicha neurona y un valor asociado a la distancia a la que se
ha activado dicho neurona (si se ha activado una neurona cercana y se ha propragado). Actualizar esto desde la GPU requiere
de exclusión mutua entre los hilos que acceden a los datos de la misma neurona.

Esta versión es mucho menos costosa computacionalmente que la propuesta en la versión on-line aunque los resultados difieren.
"""

"""
Implementación BATCH SOM
"""
import cupy as cp
from numba import cuda
import math

"""
Kernel para calcular la distancia euclídea.
Cada hebra computa la distancia entre una muestra y una neurona
"""
@cuda.jit
def batch_euclidean_distance(samples, weights, distances):
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
  


        
"""
Cada hebra se encarga de actualizar el vecindario alrededor de una BMU
La actualización se guarda en una estructura auxiliar antes de meterla en el array de pesos.

Tengo 3 opciones:
- Ejecutarlo en la CPU directamente (quizás es la mejor, pendiente de testear). El problema de esto es perder el paralelismo
y el overhead que pueda haber de traspasar bmu a host y pesos de device a host y luego de vuelta a device.

- Añadir una estructura que me permita evitar los accessos simultáneos a una posición. 
  En el caso que estoy probando de las caras, la alta dimensión 4096 floats, hace que esto si
  el mapa es relativamente grande no se viable realmente pues aumenta demasiado la complejidad espacial
  4096 * 400 ejemplos * 400 neuronas * 4 bytes = 2,5 GB!!
- La otra opción es utilizar las funciones atómicas, esta solución no es mala, especialmente en tarjetas
recientes (compute capability 6.0 en adelante), en tarjetas más antiguas puede resultar en una operación
muy cara. Esta es la que está planteada ahora mismo.
"""
@cuda.jit
def prepare_update(bmu_row, bmu_col, samples, num, den, nrows, ncols, sigma):
    idx = cuda.grid(1)
    if idx < bmu_row.size:
        my_row = bmu_row[idx]
        my_col = bmu_col[idx]
        
        init_row = max(0, my_row - int(sigma))
        finish_row = min(nrows, my_row + int(sigma) + 1)
        
        init_col = max(0, my_col - int(sigma))
        finish_col = min(ncols, my_col + int(sigma) + 1)
        
    
        for i in range(init_row, finish_row):
            for j in range(init_col, finish_col):
                dist = (j - my_col) * (j - my_col) + (i - my_row) * (i - my_row)
    
                if dist <= sigma:
                    hck = math.exp(-(dist)/(2 * sigma))
                    cuda.atomic.add(den, i*ncols+j, hck)
                    d = samples.shape[1]
                    for k in range(d):
                        cuda.atomic.add(num, i*ncols*d + j*d +k, hck * samples[idx, k])
     
        
        
"""
Finalizar actualización
"""        
@cuda.jit
def finish_update(weights, num, den):
    idx = cuda.grid(1)
    if idx < den.size:
        nrows, ncols, d = weights.shape
        row = idx // ncols
        col = idx % ncols
        if den[row * ncols + col] != 0:
            for k in range(d):
                weights[row, col, k] =  num[row*ncols*d + col*d +k] / den[row * ncols + col]
      
        cuda.syncthreads()
       
    
        for k in range(d):
            num[row*ncols*d + col*d +k] = 0
            
        den[row * ncols + col] = 0



"""
Añadir tasa de aprendizaje?
"""                
def batch(samples, rows, cols, iters, sigma=1, tau=10, seed=0, tpb=128):
    def update_sigma(iteration, sigma, tau):
        return max(0.1,math.exp(-iteration / tau) * sigma)
    
    # 1. Preparamos los elementos en el dispositivo
    # 1.1 Semilla de aleatorios y matriz de pesos
    cp.random.seed(seed)
    n, d = samples.shape
    weights = cp.random.ranf((rows, cols, d), dtype=cp.float32)
    # 1.2 Muestras.                         
    d_samples = cp.array(samples)
   
    # 1.3 Auxiliares para actualizar los pesos.
    num = cp.zeros(rows * cols * d, dtype=cp.float32)
    den = cp.zeros(rows * cols, dtype=cp.float32)
    
    # 1.4 Para almacenar las distancias euclídeas
    distances = cp.empty((n, rows * cols), dtype=cp.float32)                        
    
    # 1.5 Números de bloques
    distblocks = (n * rows * cols) // tpb + 1    
    samplesblocks = n // tpb + 1
    weightsblocks = (rows * cols) // tpb + 1
    # 2 Inicializamos el bucle del algoritmo.
    for my_iter in range(iters):
        # 2.1 Cada muestra calcula sus distancia euclídeas con todas las neuronas.
        batch_euclidean_distance[distblocks, tpb](d_samples, weights, distances)
        
        # 2.2 Obtenemos la BMU para todas las muestras aplicando una reducción 
        # a todas las filas
        bmu = cp.argmin(distances, axis=1)
        bmu_row = bmu // cols
        bmu_col = bmu % cols
        
        # 2.3 Actualizar pesos
        prepare_update[samplesblocks, tpb](bmu_row, bmu_col, d_samples, num, den, rows, cols, sigma * sigma)
        finish_update[weightsblocks, tpb](weights, num, den)
        # 2.4 Actualizar sigma para la siguiente iteración
        sigma = update_sigma(my_iter, sigma, tau)
        # print(sigma)
    return weights.get()
    
    


# In[3]:


from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
import time
inicio = time.time()
weights = batch(faces['data'], 20,20, 400, sigma=10, tau=100, seed=5)
fin = time.time()
print(f'{fin-inicio}')


# In[71]:


import matplotlib.pyplot as plt
import numpy as np
matrix_side = 20
W = weights.reshape((matrix_side,matrix_side, faces['data'].shape[1]))
pattern_length = faces['data'].shape[1]
pattern_width = pattern_height = int(np.sqrt(pattern_length))

matrix_w = np.empty((matrix_side * pattern_height, matrix_side * pattern_width))
matrix_w *= 255


for i in range(matrix_side):
    for j in range(matrix_side):
        matrix_w[i * pattern_height:i * pattern_height + pattern_height, 
j * pattern_height:j * pattern_height + pattern_width] = W[i, j].reshape((pattern_height, pattern_width)) * 255.0
fig, ax = plt.subplots(figsize=(20,20))

ax.matshow(matrix_w.tolist(), cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
  
plt.show()


# In[39]:




