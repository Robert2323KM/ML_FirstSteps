# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 22:17:12 2023

@author: rober
"""

## K-Means 

## Importar las librerías de trabajo 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

## Cargar los datos con pandas 
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values 

## Método del codo para averiguar el número óptimo de clusters 
from sklearn.cluster import KMeans 
wcss = [] 
for i in range (1, 11): # 10 primeros segmentos 
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10,random_state = 0) #Inicialización no aleatoria. 
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_) #el parámetro que trae la suma de los cuadrados de las distancias 

plt.plot(range(1,11), wcss) 
plt.title("Método del codo") 
plt.xlabel("Número de Clusters") 
plt.ylabel("WCSS(k)") 
plt.show() ## Número óptimo de K es 5 

## Aplicar el método de K-Means para segmentar el data set 
kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state= 0) 
y_kmeans = kmeans.fit_predict(X)

## Visualización de los clusters 
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = "magenta", label = "Cluster 4")
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = "gray", label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = "yellow", label = "Baricentors") 
plt.title("Cluster de clientes") 
plt.xlabel("Ingresos anuales (miles de USD)")
plt.ylabel("Puntuación de Gastos (1-100)") 
plt.legend() 
plt.show()

