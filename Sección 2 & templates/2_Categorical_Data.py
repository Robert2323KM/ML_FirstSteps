# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:37:40 2022

@author: rober
"""

## Plantilla de Pre-Procesado - Datos categóricos

import numpy as np 
## Importar una sublibrería
import matplotlib.pyplot as plt 
import pandas as pd 

## Importar el data set 
dataset = pd.read_csv('Data.csv') 

## Variable independiente
X = dataset.iloc[:,:-1].values                      ##localizar filas y 
                                                    ##columnas por posición 
                                                    ##(i de index loc de 
                                                    ##localization)

## Variable dependiente 
y = dataset.iloc[:,3].values                        ##Es minúscula porque no es


## Codificar datos categóricos 
## Tendremos que importar la librería sklearn
from sklearn.preprocessing import LabelEncoder 
## Primero iremos por la matriz de las variables independientes 
labelencoder_X = LabelEncoder()                     ##No necesita parámetros
X[:,0] = labelencoder_X.fit_transform(X[:,0]) 
'''
El tema es que, debido a que se toma como una variable númerica ordinal y no 
nominal, entonces se pueden presentar inconsistencias por el número en la 
columna 0. Por lo tanto, se importará una nueva librería que ayudará con esto.
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float) 
labelencoder_y = LabelEncoder()                     ##No necesita parámetros
y = labelencoder_y.fit_transform(y) 
                            