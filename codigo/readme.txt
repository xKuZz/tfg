Esta carpeta contiene el software asociado al proyecto planteado.


Software utilizado para el proyecto:
------------------------------------
- SDK CUDA 10.0 [necesario un dispositivo gráfico de NVIDIA compatible con CUDA].
- Python (Anaconda) 3.6.8 con los siguientes paquetes instalados de PyPy:
	- Para todos los algoritmos, Numba y NumPy.
	- Para la implementación del árbol de decisión, CuPy.
	- Para el experimento de las caras de Olivetti, matplotlib y scikit-learn. [el último para descargar el dataset]
	- Para el experimento del árbol de decisión sobre Spambase y Magic, pandas. [sólo para mostrar mejor el Notebook].
	- Todos los experimentos han sido ejecutados en Notebooks por lo que es recomendable utilizar el paquete "jupyter" para reproducir los experimentos realizados.

- Spark 2.4.0 con Hadoop 2.7.3

Contenido de los archivos:
--------------------------
El contenido de los Notebooks está destinado a ser utilizado combinando Python con Spark.

Los archivos profile_*.py están destinados para ser usados con el profiler con la orden:
	nvprof python <archivo>.py

Listado de archivos:
[Implementaciones]
- utils.py, contiene las primitivas más complejas utilizadas (reducciones y scan).
- som.py, contiene toda la implementación realizada sobre el mapa autoorganizado.
- decisiontree_gpu.py, contiene toda la implementación realizada sobre el árbol de decisión.

[Experimentos de profiling]
- profile_som.py, contiene el código para ejecutar sobre el profiler para el mapa autoorganizado.
- profile_tree.py, contiene el código para ejecutar sobre el profiler para el árbol de decisión.

[Experimentos de pruebas]
- som_olivetti.ipynb, es el Notebook que desarrolla el experimento de las caras de Linetti para comprobar el correcto funcionamiento de las implementaciones.
- SOM_BIG_DATA.ipynb, es el Notebook que desarrolla el experimento del mapa autoorganizado con SUSY.
- experimento_arbol_smalldata, es el Notebook que evalúa el entrenamiento de un árbol de decisión con MAGIC y SPAMBASE.
- ARBOL_BIG_DATA.ipynb, es el Notebook que desarrolla el experimento del random forest con SUSY.

----------------------------------------------
Los conjuntos de datos MAGIC, SPAMBASE Y SUSY pueden ser descargados de:
https://archive.ics.uci.edu/ml/index.php