import numpy as np
import decisiontree_gpu as dtree
import time
if __name__ == '__main__':
	inicio = time.time()
	magic = np.genfromtxt('../datasets/magic04.data', delimiter=',', dtype=np.float32)
	arbol = dtree.train_tree(magic)
	fin = time.time()
	print('Tiempo total:', fin-inicio)
