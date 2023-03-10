# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 20:53:08 2022

@author: rober
"""

##Plantilla de Regresión

## Librerías
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

## Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:,1:2].values                     
y = dataset.iloc[:,2].values 

## Dividir el dataset en conjunto de entrenamiento y de testing
"""
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, 
                                                  random_state=0) """ 
## Escalado de variables 
""" 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

## Ajustar la regresión con el dataset 
## Crear aquí nuestro modelo de regresión 

## Predicción de nuestros modelos 
y_pred = regression.predict(([[6.5]]))

## Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X),max(X), 0.1) 
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = "red") 
plt.plot(X_grid, regression.predict((X_grid)), color = "blue") 
plt.title("Modelo de Regresión") 
plt.xlabel("Sueldo (en $)") 
plt.show() 
