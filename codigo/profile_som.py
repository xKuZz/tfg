import numpy as np
import som
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
"""
En este ejemplo en puro Python simulamos una iteración del algoritmo en Spark.
Para empezar, generamos la matriz de pesos de forma aleatoria.
Segundo, realizamos el proceso que se hace en cada nodo 10 veces para simular
las 10 particiones del RDD usadas.
Por último, lanzamos el kernel que junta los resultados parciales.
"""
if __name__ == '__main__':
    np.random.seed(0)
    N = 1000000
    d = 18
    multiprocessors = 10
    rows = 10
    cols = 10
    sigma_squared = 10
    tpb = 128
    partitions = [np.float32(np.random.ranf((N//multiprocessors, d))) for i in range(multiprocessors)]

    # 1. Cálculo de pesos iniciales
    d_weights = cuda.device_array((rows, cols ,d), np.float32)
    rng_states = create_xoroshiro128p_states(rows * cols * d, seed=3)
    som.rand_weights[(d_weights.size) // tpb + 1, tpb](rng_states, d_weights)
     
    weights = d_weights.copy_to_host()

    # 2. Cálculos de pesos parciales
    out = [som.gpu_work_iter(weights, sigma_squared)(data) for data in partitions]
    numParts = len(out) // 2
    out = list(map(np.concatenate, out))
    partials = np.concatenate(out)

    # 3. Cálculo de psos finales
    som.finish_update[rows * cols // tpb + 1, tpb](weights, partials, numParts)