import numpy as np
import cupy as cp
import numba
from numba import cuda
import math
import _somkernels

class SOM:
    def __init__(self, rows, cols, d):
        self._rows = rows
        self._cols = cols
        self._d = d
        self._tpb = 128
        self._seed = None
        self._weights = None

    @property
    def rows(self):
        return self._rows
    
    @property
    def cols(self):
        return self._cols

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset
    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def tpb(self):
        return self._tpb

    @tpb.setter
    def tpb(self, tpb):
        self._tpb = tpb
    

    def train_online_cpu(self, max_iters, nsamples=None,
        smooth_iters=None, sigma_0=10, sigma_f=0.01, eta_0=1, 
        eta_f=0.1, tau=400):
        np.random.seed(self._seed)
        # 1. Inicializamos los pesos
        if self._weights is None:
            self._weights = np.random.ranf((self._rows, self._cols, self._d))

        if nsamples is None:
            nsamples = len(self._dataset)

        # 2. Bucle del algoritmo
        indexes = np.arange(len(self._dataset))
        for t in range(max_iters):
            init = time.time()
            print(f'Iteracion {t}')
            # 2.1 Actualizamos los parámetros sigma y eta
            if smooth_iters is None or t < smooth_iters:
                sigma = sigma_0 * math.exp(-t/tau)
                eta = eta_0 * math.exp(-t/tau)
            else:
                sigma, eta = sigma_f, eta_f

            # Seleccionamos un subconjunto de muestras a evaluar
            current_indexes = np.random.choice(indexes, nsamples)

            for current in current_indexes:
                # 2.2 Calculamos las distancias entre una muestra y las neuronas
                distances = np.linalg.norm(self._dataset[current] - self._weights, axis=2)

                # 2.3 Buscamos la BMU
                bmu_index = np.argmin(distances)
                bmu_row, bmu_col = bmu_index // self._cols, bmu_index % self._cols

                # 2.4 Actualizamos la región alrededor de dicha BMU
                for i in range(self._rows):
                    for j in range(self._cols):
                        d_f = (j - bmu_col) * (j - bmu_col)
                        d_f += (i - bmu_row) * (i - bmu_row)

                        sigma_squared = sigma * sigma
                        if d_f <= sigma_squared:
                            d_f = math.exp(-d_f/sigma_squared)
                            self._weights[i,j] += eta * d_f * \
                            (self._dataset[current] - self._weights[i,j])

            
    def train_batch_cpu(self, max_iters, smooth_iters=None, 
        sigma_0=10, sigma_f=0.01, tau=400):
        np.random.seed(self._seed)
        # 1. Inicializamos los pesos
        if self._weights is None:
            self._weights = np.random.ranf((self._rows, self._cols, self._d))

        if nsamples is None:
            nsamples = len(self._dataset)

        indexes = np.arange(len(self._dataset))
        # 2. Bucle del algoritmo
        for t in range(max_iters):
            print(f'Iteración {t}')
            current_indexes = np.random.choice(indexes, nsamples)
            # 2.1 Actualizamos los parámetros sigma y eta
            if smooth_iters is None or t < smooth_iters:
                sigma = sigma_0 * math.exp(-t/tau)
            else:
                sigma = sigma_f

            # 2.2 Calculamos la distancia y la BMU entre todas las muestras y los pesos.
            bmus = np.array([np.argmin(np.linalg.norm(self._dataset[c] - self._weights, axis=2)) for c in current_indexes])
            bmu_rows, bmu_cols = bmus // self._cols, bmus % self._cols

            nums = np.zeros(self._weights.shape)
            dens = np.zeros((self._rows, self._cols))

            # 2.3 Generamos los nuevos pesos.
            sigma_squared = sigma * sigma
            for x, (row, col) in enumerate(zip(bmu_rows, bmu_cols)):
                for i in range(self._rows):
                    for j in range(self._cols):
                        hck = (j - col) * (j - col) + (i - row) * (i - row)
                
                        if hck <= sigma_squared:
                            hck = math.exp(-hck/(2 * sigma_squared))
                            dens[i, j] += hck
                            nums[i, j, :] += hck * self._dataset[x]
            
            for i in range(self._rows):
                for j in range(self._cols):
                    if dens[i, j] != 0:
                        self._weights[i, j] = nums[i, j] / dens[i, j]
            

    def train_online_cuda(self, max_iters, nsamples=None, smooth_iters=None, 
        sigma_0=10, sigma_f=0.01, eta_0=1, eta_f=0.1, tau=400):
        np.random.seed(self._seed)
        cp.random.seed(self._seed)

        # 1. Inicializamos los pesos
        if self._weights is None:
            self._weights = cp.random.ranf(self._rows * self._cols * self._d, dtype=cp.float32)

        if nsamples is None:
            nsamples = len(self._dataset)

        # 1.1 Inicializamos el resto de variables de memoria para el dispositivo
        sm_size = self._dataset[0].size * self._dataset[0].dtype.itemsize
        blocks = self._rows * self._cols // self._tpb + 1
        dsamples = cp.array(self._dataset.flatten(), cp.float32)
        distances = cp.empty(self._rows * self._cols, cp.float32)

        # 2. Bucle del algoritmo
        indexes = np.arange(len(self._dataset))
        for t in range(max_iters):
            print(f'Iteracion {t}')
            # 2.1 Actualizamos los parámetros sigma y eta
            if smooth_iters is None or t < smooth_iters:
                sigma = sigma_0 * math.exp(-t/tau)
                eta = eta_0 * math.exp(-t/tau)
            else:
                sigma, eta = sigma_f, eta_f

            # Seleccionamos un subconjunto de muestras a evaluar
            current_indexes = np.random.choice(indexes, nsamples)
            for current in current_indexes:
                # 2.2 Calculamos las distancias entre una muestra y las neuronas
                _somkernels.euclidean_distance[blocks, self._tpb, 0, sm_size](current, dsamples, self._weights, distances, self._d)

                # 2.3 Buscamos la BMU
                min_index = int(cp.argmin(distances))
                bmu_row, bmu_col = min_index // self._cols, min_index % self._cols
                sigma_squared = sigma * sigma
                # 2.4 Actualizamos la región alrededor de dicha BMU
                _somkernels.bmu_update[blocks, self._tpb](current, dsamples, self._weights, self._d, bmu_row, bmu_col, self._cols, eta, sigma_squared)


        self._weights = self._weights.get().reshape((self._rows, self._cols, self._d))

    def train_batch_cuda(self, max_iters, smooth_iters=None, sigma_0=10, 
        sigma_f=0.01, tau=400):
        # 1.1 Semilla de aleatorios y matriz de pesos
        cp.random.seed(self._seed)
        self._weights = cp.random.ranf((self._rows, self._cols, self._d), dtype=cp.float32)
        # 1.2 Muestras.                         
        dsamples = cp.array(self._dataset, dtype=cp.float32)
       
        # 1.3 Auxiliares para actualizar los pesos.
        num = cp.zeros(self._rows * self._cols * self._d, dtype=cp.float32)
        den = cp.zeros(self._rows * self._cols, dtype=cp.float32)
        
        # 1.4 Para almacenar las distancias euclídeas
        distances = cp.empty((len(self._dataset), self._rows * self._cols), dtype=cp.float32)                        
        
        # 1.5 Números de bloques
        distblocks = (len(self._dataset) * self._rows * self._cols) // self._tpb + 1    
        samplesblocks = len(self._dataset) // self._tpb + 1
        weightsblocks = (self._rows * self._cols) // self._tpb + 1

        # 2. Bucle del algoritmo
        for t in range(max_iters):
            print(f'Iteración {t}')
            # 2.1 Actualizamos los parámetros sigma y eta
            if smooth_iters is None or t < smooth_iters:
                sigma = sigma_0 * math.exp(-t/tau)
            else:
                sigma = sigma_f

            # 2.2 Calculamos la distancia y la BMU entre todas las muestras y los pesos.
            _somkernels.batch_euclidean_distance[distblocks, self._tpb](dsamples, self._weights, distances)
            bmu = cp.argmin(distances, axis=1)
            bmu_row = bmu // self._cols
            bmu_col = bmu % self._cols

            # 2.3 Generamos los nuevos pesos.
            sigma_squared = sigma * sigma
            _somkernels.prepare_update[samplesblocks, self._tpb](bmu_row, bmu_col, dsamples, num, den, self._rows, self._cols, sigma_squared)
            _somkernels.finish_update[weightsblocks, self._tpb](self._weights, num, den)
        
        self._weights = self._weights.get().reshape((self._rows, self._cols, self._d))

    def quantification_error_cpu(self):
        error = 0.0
        for sample in self._dataset:
            error += np.min(np.linalg.norm(sample - self._weights,axis=2))
            
        return error / len(self._dataset)

    def topography_error_cpu(self):
        error = 0
        for sample in self._dataset:
            distances = np.linalg.norm(sample - self._weights, axis=2)

            bmus = np.argsort(distances, axis=None)[:2]
            print(bmus)
            bmu_rows = bmus // self._cols
            bmu_cols = bmus % self._cols
            print(bmu_rows, bmu_cols)

            if abs(bmu_rows[1] - bmu_rows[0]) == 1 and abs(bmu_cols[1] - bmu_cols[0]) == 1:
                error += 1

        return error / len(self._dataset)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_olivetti_faces
    import time
    faces = fetch_olivetti_faces()

    som = SOM(20,20,4096)
    som.dataset = faces['data']
    som.seed = 20

    inicio = time.time()
    # som.train_online_cpu(max_iters=400, nsamples=400, smooth_iters=100)
    som.train_batch_cuda(max_iters=400, smooth_iters=100, sigma_0=15)

    fin = time.time()
    print(f'El algoritmo ha tardado {fin - inicio} segundos.')
    print(f'Error cuantificación {som.quantification_error_cpu()}')
    print(f'Error topográfico {som.topography_error_cpu()}')

    W = som.weights
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