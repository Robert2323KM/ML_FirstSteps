# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:53:03 2023

@author: rober
"""

## Muestreo Thompson 

import numpy as pd 
import matplotlib.pyplot as plt 
import pandas as pd 

## Cargar el dataset 
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

## Algoritmo de Muestro Thompson 
import random ##import math 
N = 10000 
d = 10 
number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d
ads_selected = [] 
total_reward = 0 
for n in range (0, dataset.shape[0]): 
    max_random = 0 ##max_upper_bound = 0 El que tenga mayor probabilidad de éxito  
    ad = 0 
    for i in range(0, dataset.shape[1]):
        ##Generar 10 números al azar para cada anuncio 
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random: 
          max_random = random_beta 
          ad = i 
    ads_selected.append(ad)
    reward = dataset.values[n, ad] 
    if reward == 1: 
        number_of_rewards_1[i] = number_of_rewards_1[i] + 1 
    else: 
        number_of_rewards_0[i] = number_of_rewards_0[i] + 1
    total_reward = total_reward + reward 
    
## Histograma de resultados 
plt.hist(ads_selected) 
plt.title("Histograma de anuncios") 
plt.xlabel("ID del anuncio") 
plt.ylabel("Frecuencia de visualización del anuncio") 
plt.show()