# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:17:02 2023

@author: rober
"""

##Artificial Neural Networks(Deep Learning)
##Install Theano 
    ## pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
    ## Developed by Monreal's workers
##Install Tensorflow & Keras 
    ##(Conda): 
        ##conda install -c conda-forge keras (Tensorflow)
    ## Developed by Google's team. 
    ## Keras was made because Fransua Chale, Google ML scientist. 
    ## A lib for dummies in ML. 
    
##--------------------------------------------------------------------#
##Part 1 - Data preprocessing 
##--------------------------------------------------------------------#

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

##Dataset was given to identify some reasons why customers have left. 
dataset = pd.read_csv('Churn_Modelling.csv') 
X = dataset.iloc[:,3:-1].values 
y = dataset.iloc[:,-1].values 

##X = dataset.iloc[:, 3:13].values
##y = dataset.iloc[:, 13].values
 
##Categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
labelencoder_X_1 = LabelEncoder() ##Countries column
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) 
labelencoder_X_2 = LabelEncoder() ##Gender column
X[:,2] = labelencoder_X_1.fit_transform(X[:,2]) 
labelencoder_X_3 = LabelEncoder() ##Countries column
X[:,3] = labelencoder_X_1.fit_transform(X[:,1]) 
##Turning new numeric columns into different columns
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float) 
##Neither Spain neither German then French, so it's good to delete column 0 (French column) 
##in order to avoid multicolonality
X = X[:,1:]

##Getting Test and training set 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, 
                                                    random_state=0) 

## standardization of variables
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)  


##Part 2 - Building the ANN 
import keras 
from keras.models import Sequential ##for initializing NN's parameters
from keras.layers import Dense 

##Initialize the ANN 
##Creating a sucesive NN 
    ##Defining the class number 
classifier = Sequential() 

##Adding input layers and the first hiden layer
    ##Adding one by one each layer 
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11)) ##Dense is something like "Synapse zone". Units is the amout of hiden layers. Half between the amount of Input and output layers. We are using input_dim (input dimension) instead of input_shape because it is a Tensor which we would need to specify sample size and data dimension (in this case 11). Units -> Hiden layers / input_dim -> Input layers. 
##Adding a new hiden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu")) ##second layer knows that last one had 6 output layers so we don't need to specify "input_dim = 6".
##Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid")) ##output layer, so we only need 1 output layer (units parameter). We need more a probability activation function so sigmoid function is good.

##NN Compilation 
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) ##optimizer -> Algoryth which finds the optimal solution sets/weights. Descendent gradient, Adam Optimizer, Stocastic descendent gradient

##Adjusting the NN to the training set 
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

##Part 3 - Evaluating the model and calculating final predictions 
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
##Confussion Matrix 
from sklearn.metrics import confusion_matrix ##Not a class but a function 
cm = confusion_matrix(y_test, y_pred) ## cm = confusion matrix

## Representación gráfica de los resultados del algoritmo en el Conjunto de Entramiento 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


