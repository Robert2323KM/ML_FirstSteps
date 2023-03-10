# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

##Convolutional Neural Network 

##Install Theano
##pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git 
    ##if upper command doesn't work, try this (anaconda prompt): 
        ##pip uninstall protobuf 
        ##pip uninstall tensorflow 
        ##pip install protobuf 
        ##pip install tensorflow
##Intall Tensorflow & Keras 
##conda install -c conda-forge keras

##Part 1 - building the CNN model 
from keras.models import Sequential ##initializes cnn and random weights 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense ##Dense for synopsis 

##Initialize the CNN 
classifier = Sequential()

##Step 1 - Convolution 
classifier.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (64, 64, 3), activation = "relu")) ##Kernel is the amount of feature detectors and their size (3x3) 

##Step 2 - Max Pooling 
classifier.add(MaxPooling2D(pool_size = (2,2))) ##first attemp try with 2,2 if wanna a better performance, try to explore with power of 2 values

##Part 3 - Improving the CNN
##Second Convolution and Max Pooling layer 
classifier.add(Conv2D(filters = 32, kernel_size = (3,3), activation = "relu")) 
classifier.add(MaxPooling2D(pool_size = (2,2))) 

classifier.add(Conv2D(filters = 32, kernel_size = (3,3), activation = "relu")) 
classifier.add(MaxPooling2D(pool_size = (2,2))) 

##Step 3 - Flattening 
classifier.add(Flatten()) ##Getting a vector from the latest matrix so that the first input layer is able to get the input data 
 
##Step 4 - Full Connection 
classifier.add(Dense(units = 128, activation = "relu")) ## Adding hiden layers
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units= 1, activation = "sigmoid"))

##Compile CNN(Backward propagation) 
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

##Part 2 - Fitting the CNN with iamges of training set
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_dataset = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
testing_dataset = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')
print("HOLA") 
classifier.fit_generator(training_dataset, steps_per_epoch=8000/32, epochs=25, validation_data=testing_dataset, validation_steps=2000/32)

##Predict some images ;) 
##Saving the model 
classifier.save('classifier.h5')
 
##Importin the saved model 
from keras.models import load_model
import numpy as np
import cv2
 
CATEGORIES = ["Cat", "Dog"]
model = load_model('classifier.h5')
 
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
 
img = cv2.imread('dataset/test_image/cat.4010.jpg') 
img = cv2.resize(img,(64,64)) ##resize the image to fit into the model
img = np.reshape(img,[1,64,64,3])
classes = model.predict_classes(img)
 
print(classes)
print(CATEGORIES[int(classes[0][0])])