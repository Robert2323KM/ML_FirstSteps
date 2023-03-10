# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:29:50 2023

@author: rober
"""

##NPL + Classification task 

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

##...........................................................................##
##.........................Classification Algorythm..........................##
##..............................SVM Algorythm................................##

##Getting Testing and Traing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, 
                                                    random_state=0) 

##Does not need standardization because the data set has no values other 
##than 1 or 0

##Fixing SVM into Traing Set 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix 
kernels = ['linear', 'rbf', 'poly']
confusion_matrixes = [] 
for kernel in kernels: 
    classifier = SVC(kernel = kernel, C = 30) 
    classifier.fit(X_train, y_train) 
    ## Prediction with Testing set 
    y_pred = classifier.predict(X_test)
    ##Confusion Matrix
    cm = confusion_matrix(y_test, y_pred) 
    confusion_matrixes.append(cm) 
##As seen in this case, linear is the best kernel for this svm algorythm 

##Getting prediction again with linear kernel 
classifier = SVC(kernel = 'linear', C = 30) 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
cm_linear = confusion_matrix(y_test, y_pred) 

##Main variables
Accuracy = 0
Precision = 0
Recall = 0
F1_Score = 0 
TN, FP, FN, TP = 0,0,0,0
##-------------- 
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
total = TP+TN+FP+FN
Accuracy = (TP+TN)/(total)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*Precision*Recall/(Precision+Recall)
##Accuracy = 0.735 
##Precision = 0.7717391304347826
##Recall = 0.6893203883495146
##F1_Score = 0.7282051282051282 