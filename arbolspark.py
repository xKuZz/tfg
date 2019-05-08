#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cupy as cp
import math
import miarbol

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("tree").config("spark.executor.memory", "4gb").getOrCreate()
sc = spark.sparkContext


# In[2]:


"""
Para realizar este modelo cada nodo de spark genera un árbol con una porción de las muestras
durante el proceso de entrenamiento.
"""
def gpu_rf_spark(d, max_depth, min_samples_per_node, tpb):
    def _gpu_work(data):
        inp = np.asarray(list(data), dtype=np.float32)
        inp = np.reshape(inp, (inp.size // d, d))
        model = miarbol.train_decision_tree(inp, max_depth, min_samples_per_node, tpb)
        return [model]
    
    return _gpu_work



def train_random_forest(sc, dataset, ntrees=None, max_depth=10, min_samples_per_node=1, tpb=128):
    d = dataset.shape[1]
    if ntrees is None:
        rdd_data = sc.parallelize(dataset)
    else:
        rdd_data = sc.parallelize(dataset, ntrees)
        
    rdd_data = rdd_data.mapPartitions(gpu_rf_spark(d, max_depth, min_samples_per_node, tpb))
    
    return rdd_data.collect()
    
"""
Una vez entrenados una muestra puede ser evaluada 
usando todos los árboles generados y cogiendo la
clase con voto mayoritario. Esto luego se podría 
pasa a CUDA para evaluar todas las muestras a la vez
de un conjunto si queremos.
"""
def predict(sample, trees):
    counter = 0
    for tree in trees:
        counter += miarbol.predict(sample, tree)
    
    return 1 if counter > len(trees) / 2 else 0

def evaluar(dataset, trees):
    aciertos = 0
    for sample in dataset:
        etiqueta = predict(sample, trees)
        if int(etiqueta) == int(sample[-1]):
            aciertos += 1
        
    return aciertos / dataset.shape[0]


# In[3]:


dataset = miarbol.getdataset_magic()
# Los conjuntos de datos vienen completamente ordenados por clases
# Los barajamos de forma aleatoria para que a cada árbol le puedan
# corresponder elementos de la clase positiva como de la clase negativa.
np.random.seed(40)
np.random.shuffle(dataset)


# In[4]:


trees = train_random_forest(sc, dataset, ntrees=10, max_depth=6)


# In[5]:


accuracy = evaluar(dataset, trees)
print('El porcentaje de acierto es del', accuracy * 100 , '%')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




