# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:12:19 2022

@author: robert
"""

## Plantilla de pre-procesado 

## Cómo importar las librerías 

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
                                                    ##una matriz sino un array
                                                    
## Tratamiento de los NaN 
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=
                        None, verbose=0, copy=True, add_indicator=False);
                                                    ## 0 Col, 1 Rows
                                                    
#here fit method will calculate the required parameters (In this case mean)
#and store it in the impute object                                                    
imputer.fit(X[:,1:3])                               ##El último valor no es 
                                                    ##tomado 
                                                    
#imputer.transform will actually do the work of replacement of nan with mean.
#This can be done in one step using fit_transform
X[:,1:3] = imputer.transform(X[:,1:3])              ##transform se encarga de 
                                                    ##que los datos sean 
                                                    ##sustituidos 

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

## Dividir el data set en conjunto de entrenamiento y de testing 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, 
                                                    random_state=0) 
                                                    ## es para 
                                                    ##dar un ordenamiento azar 

## Escalado de variables (estandarización)
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)               ##El primero calcula los 
X_test = sc_X.transform(X_test)                     ##coeficientes y ajusta los
'''Si tienes el conjunto entero. En la'''           ## datos, el segundo solo
'''realidad cuando te llegan nuevos datos'''        ##ajusta los datos con los 
'''usas el conjunto de test para simular '''        ##coeficientes que ha
'''las mismas transformaciones que harías'''        ##calculado antes 
'''con esos datos de ahí que lo haga así.'''                                                 

## Escalado de valores


 
                                                    
                                                     
                                                    



