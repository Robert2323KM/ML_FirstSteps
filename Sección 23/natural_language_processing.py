# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:07:14 2023

@author: rober
"""

## Natural Language Processing 

## Importar librerías 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

## Importar el dataset 
## el separador estándar no debería ser coma, porque es lenguaje, mejor el tabulador. 
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t", quoting = 3) 

## Limpieza de texto 
import re 
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer ##eliminará las conjugaciones de una determinada palabra
corpus = [] ##corpus es colección de textos que sirven para cualquier tipo de algoritmo 
for i in range(0,dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) ##^ son las cosas que no quiero eliminar
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)##juntar las palabras
    corpus.append(review)

##Crear el Bag of Words 
from sklearn.feature_extraction.text import CountVectorizer ##Importante. Crea las variables en columnas. Generando la matriz dispersa 
cv = CountVectorizer(max_features = 1500) ##word to vect 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

##Implementación de Naive Bayes------------------------------------------------------------ 

## Dividir el data set en conjunto de entrenamiento y de testing 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, 
                                                    random_state=0) 
                       
## Escalado de variables (estandarización)
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)  

## Ajustar el clasificador en el Conjunto de Entrenamiento 
from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB() 
classifier.fit(X_train, y_train)
    
## Predicción de los resultados con el Cojunto de Testing 
y_pred = classifier.predict(X_test)

## Elaborar una matriz de confusión 
    ##Para validar que las predicciones sean útiles y potentes 
from sklearn.metrics import confusion_matrix ##No es una clase sino una función 
cm = confusion_matrix(y_test, y_pred) ## cm = confusion matrix ###cm:10
