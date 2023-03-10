# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:43:45 2022

@author: rober
"""

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


 
                                                    
                                                     
                                                    



