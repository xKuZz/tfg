import numpy as np
import decisiontree_gpu as dtree

if __name__ == '__main__':
	
	magic = np.genfromtxt('../datasets/magic04.data', delimiter=',', dtype=np.float32)
	dtree.train_tree(magic)
	
