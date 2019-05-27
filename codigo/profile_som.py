import numpy as np
import som

if __name__ == '__main__':
	np.random.seed(0)
	N = 100000
	d = 18
	rows = 10
	cols = 10
	sigma_squared = 10
	data = np.float32(np.random.ranf((N * d)))
	weights = np.float32(np.random.ranf(rows * cols * d))

	nums, denums = som.gpu_work_iter(d, rows, cols, weights, sigma_squared)(data)