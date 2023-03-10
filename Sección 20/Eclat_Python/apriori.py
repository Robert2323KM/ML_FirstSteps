# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:54:22 2023

@author: rober
"""

## Apriori 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

## Importar el data set 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) 
## Se hace una lista de listas para que tome 1 transacción grande por cada registro
transactions = [] 
for i in range(0, dataset.shape[0]): 
    transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])]) ##El .values es para no guardar info de filas y columnas. El str es para que se guarden entre comillas 

## Entrenar el algoritmo de Apriori 
from apyori import apriori 
association_rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2) ##min_length para evitar dar reglas de asociación de 1 ítem, por ejemplo / 0.0028 es 3 compras por día 3x7(días)/7500
## 0.003 ->21 veces por semana. Un 20% de nivel de confianza en donde se compre el primer ítem. Lift 3 

##Visualización de los resultados 
results = list(association_rules) 
results[0]


# visualizations ---------------------------------------------------------
plt.plot(association_rules, method = "graph", engine = "htmlwidget")

