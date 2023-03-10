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
