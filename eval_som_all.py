"""
Experimento de evaluación del modelo SOM
"""
import numpy as np
import som
import time

np.random.seed(6)
dataset = np.random.ranf((10000, 100))

mysom = som.SOM

repeat = 10
topograhpy_errors = np.empty((5, repeat))
indexes = np.arange(10000)
for mysamples in [1000, 3000, 5000, 7000, 10000]:
    for k in range(repeat):
        print('N:', mysamples, "Repetición", k)
        mysom = som.SOM(7, 8, 100)
        selected = np.random.choice(indexes, mysamples)
        mysom.dataset = np.array(dataset[selected], dtype=np.float32)

        # CPU ONLINE
        start = time.time()
        mysom.train_online_cpu(max_iters=50, smooth_iters=25, nsamples=mysamples, eta_0=1, eta_f=0.01, sigma_0=5, sigma_f=0.1, tau=100)
        end = time.time()
        print('Tiempo CPU online:', end - start)

        # CPU BATCH
        mysom.weights = None
        start = time.time()
        mysom.train_batch_cpu(max_iters=50, smooth_iters=25, sigma_0=5, sigma_f=0.1, tau=100)
        end = time.time()
        print('Tiempo CPU batch:', end - start)


        # CUDA ONLINE
        mysom.weights = None
        start = time.time()
        mysom.train_online_cuda(max_iters=50, smooth_iters=25, nsamples=mysamples, eta_0=1, eta_f=0.01, sigma_0=5, 
                               sigma_f=0.1, tau=100)
        end = time.time()
        print('Tiempo GPU online:', end - start)
        print('Error de cuantificación online:', mysom.quantification_error())
        print('Error topográfico online:', mysom.topography_error())

        # CUDA BATCH
        mysom.weights = None
        start = time.time()
        mysom.train_batch_cuda(max_iters=50, smooth_iters=25, sigma_0=5, sigma_f=0.1, tau=100)
        end = time.time()
        print('Tiempo GPU batch:', end - start)
        print('Error de cuantificación batch:', mysom.quantification_error())
        print('Error topográfico bach:', mysom.topography_error())