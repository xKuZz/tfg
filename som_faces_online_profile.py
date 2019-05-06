import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import som
import time

if __name__ == '__main__':
    mysom = som.SOM(20, 20, 4096)
    mysom.dataset = fetch_olivetti_faces()['data']
    mysom.seed = 20

    start = time.time()
    mysom.train_online_cuda(max_iters=400, nsamples=400, smooth_iters=100, sigma_0=15, 
                           sigma_f=0.01, tau=400)
    end = time.time()
    print('Tiempo (s):', end - start)
    print('Error de cuantificación:', mysom.quantification_error())
    print('Error topográfico:', mysom.topography_error())

    image_size = 4096
    width = height = int(np.sqrt(image_size))
    output = np.empty((20 * height, 20 * width))

    for i in range(20):
        for j in range(20):
            output[i * width:(i+1) * width, j * width:(j + 1) * width] = mysom.weights[i,j].reshape((height, width)) * 255.0
    
    fig, ax = plt.subplots(figsize=(12,12))
    ax.matshow(output.tolist(), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
      
    plt.show()