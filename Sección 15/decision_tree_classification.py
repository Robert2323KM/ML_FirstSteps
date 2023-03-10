# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 12:38:04 2023

@author: rober
"""

## Clasificación con árboles de Decisión 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('Social_Network_Ads.csv') 
X = dataset.iloc[:,[2,3]].values 
y = dataset.iloc[:,-1].values  

## Dividir el data set en conjunto de entrenamiento y de testing 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, 
                                                    random_state=0) 
                       
## Escalado de variables (estandarización) NO SE NECESITA PORQUE NO USA DISTANCIAS!!
##from sklearn.preprocessing import StandardScaler 
##sc_X = StandardScaler() 
##X_train = sc_X.fit_transform(X_train) 
##X_test = sc_X.transform(X_test)  

## Ajustar el Árbol de Decisión en el Conjunto de Entrenamiento 
from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)  
classifier.fit(X_train, y_train) 

## Predicción de los resultados con el Cojunto de Testing 
y_pred = classifier.predict(X_test)

## Elaborar una matriz de confusión 
    ##Para validar que las predicciones sean útiles y potentes 
from sklearn.metrics import confusion_matrix ##No es una clase sino una función 
cm = confusion_matrix(y_test, y_pred) ## cm = confusion matrix ##cm = 10 

## Representación gráfica de los resultados del algoritmo en el Conjunto de Entramiento 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1), ## ya no son step = 0.01 porque no se hizo escalado de variables
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 500)) ## ya no son step = 0.01 porque no se hizo escalado de variables
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Árbol de Decisión (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 500))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Árbol de Decisión (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
