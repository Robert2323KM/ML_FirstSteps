import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) #no hay títulos para que no tome como columna el primer dato en este caso
transactions = []
for i in range(0, 7501): #bucle de cada cliente (7500) 
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)]) #con su transacción con sus items (20) que guarde values y strings (str)
    
    
# Entrenar el algoritmo de Eclat
from apyori import apriori
rules = apriori(transactions, min_support = 0.004 , min_length = 2)
 
# Visualización de los resultados
results = list(rules)
 
results[1]