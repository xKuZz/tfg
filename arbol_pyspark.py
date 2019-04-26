"""
Cargar datasets.
Datasets seleccionados:
Spambase
Magic Gamma Telescope
"""
import numpy as np
import cmath
import time

"""
SPAMBASE
"""
def getdataset_spambase():
    return np.genfromtxt('./datasets/spambase.data', delimiter=',', dtype=np.float32)

"""
Magic Gamma Telescope
"""
def getdataset_magic():
    # 1 es Clase G y 0 es Clase H, ya modificado en el archivo que se lee por comodidad
    return np.genfromtxt('./datasets/magic04.data', delimiter=',', dtype=np.float32)


from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
import numpy as np
import time
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

spark = SparkSession.builder.master("local").appName("CUDT").config("spark.executor.memory", "4gb").getOrCreate()
sc = spark.sparkContext

def pyspark_cross_validation(dataset, k=10, depth=5, my_min=1):
    accuracy = np.zeros(k, np.float32)
    times = np.zeros(k, np.float32)
    rdd1 = sc.parallelize(dataset)
    rdd1 = rdd1.map(lambda x: [float(i) for i in x])
    rdd1 = rdd1.map(lambda x: LabeledPoint(x[-1], x[:-2]))
    for i in range(k):
        (trainingData, testData) = rdd1.randomSplit([0.9, 0.1], seed = i)
        inicio = time.time()
        model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=depth, maxBins=32)
        fin = time.time()
        times[i] = fin-inicio
        predictions = model.predict(testData.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        accuracy[i] = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(testData.count())
        
    print('Profundidad', depth, 'Velocidad', np.mean(times), 'Precisión', np.mean(accuracy))

print('DATASET SPAMBASE')
for i in range(4, 11):
    pyspark_cross_validation(getdataset_spambase(), depth=i)

print('DATASET MAGIC04')
for i in range(4, 11):
    pyspark_cross_validation(getdataset_magic(), depth=i)