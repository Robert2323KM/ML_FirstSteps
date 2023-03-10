# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:31:21 2023

@author: rober
"""

##...........................................................................##
##..............................NPL Algorythm................................##
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

## Import dataset 
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t", quoting = 3) 

## Text cleansing
import re 
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer ##Deletes conjugations
corpus = []                                ##Corpus: Text collection with clean data
for i in range(0,dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) 
    corpus.append(review)

##Creating Bag of Words 
from sklearn.feature_extraction.text import CountVectorizer ##Helps to create the sparse matrix  
cv = CountVectorizer(max_features = 1500) ##word to vect 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values 

##Getting Testing and Traing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, 
                                                    random_state=0) 

##...........................................................................##
##.........................Classification Algorythm..........................##
##.............................CART Algorythm................................##

##Fixing CART into Traing Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train,y_train)

##Does not need standardization because the data set has no values other 
##than 1 or 0

## Prediction with Testing set 
y_pred  = classifier.predict(X_test)

##Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_CART = confusion_matrix(y_test, y_pred)


