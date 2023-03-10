# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:30:32 2023

@author: rober
"""

## Clustering Jerárquico
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

## Importar los datos del centro comercial con pandas 
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values ## el .values es para seleccionar los datos y no las filas y columnas 

## Utilizar el dendograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch 
dendogram = sch.dendrogram(sch.linkage(X, method = "ward")) ##aglomerativo
plt.title("Dendograma") 
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea") 
plt.show()

## Ajustar el clustering jerárquico a nuestro conjunto de datos 
from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X) 

## Visualización de los clusters 
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s = 100, c = "magenta", label = "Cluster 4")
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s = 100, c = "gray", label = "Cluster 5")
plt.title("Cluster de clientes") 
plt.xlabel("Ingresos anuales (miles de USD)")
plt.ylabel("Puntuación de Gastos (1-100)") 
plt.legend() 
plt.show() 

