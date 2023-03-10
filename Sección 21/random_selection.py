# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:53:28 2023

@author: rober
"""

## Upper Confidence Bound (UCB)
import numpy as pd 
import matplotlib.pyplot as plt 
import pandas as pd 

## Cargar el dataset 
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

## Implementando Random Selection 
import random 
N = 10000 
d = 10 
ads_selected = [] 
total_reward = 0 
for n in range (0, N): 
    ad = random.randrange(d) 
    ads_selected.append(ad) 
    reward = dataset.values[n, ad] 
    total_reward = total_reward + reward 
    
## Visualizando los resultados - Histograma
plt.hist(ads_selected) 
plt.title("Histograma de selección de anuncios") 
plt.xlabel("Anuncio") 
plt.ylabel("Número de veces que ha sido visualizado")
plt.show() 
