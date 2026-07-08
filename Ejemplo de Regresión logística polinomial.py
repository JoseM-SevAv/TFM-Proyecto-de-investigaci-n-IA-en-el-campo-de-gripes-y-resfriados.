# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:00:01 2026

@author: chema
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("data/dataset.csv") # Metemos en un dataframe nuestro dataset
df = df.dropna() # Eliminamos los valores faltantes si los hubiese
  
 
# Seleccionamos variables predictoras y objetivo

X_df = df.drop('Valores categoricos', axis=1) 
y_df = df['Valores categoricos']

#Escalado

scaler = StandardScaler() # Objeto que hace el escalado de variables
X_scaled = scaler.fit_transform(X_df) # Calcula los parámetros de escalado y los aplica

# Dividimos en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_df, test_size=0.2, random_state=42
)

# Creamos un modelo de clasificación, lo haremos de grado 3 en este ejemplo porque la mayoría de datos tienen 3 categorías.

grado = 3
polinomio = sklearn.preprocessing.PolynomialFeatures(degree=grado)
X_pol = polinomio.fit_transform(X_train)
Y_pol = y_train

# Entrenamos el modelo con los datos estandarizados

clf = sklearn.linear_model.LogisticRegression(random_state=42)
clf.fit(X_pol, Y_pol)

# Hacemos predicciones sobre el test set

y_pred = clf.predict(X_pol)

# Ploteamos el resultado y mostramos la precisión del modelo

plot_decision_boundary(lambda x: clf.predict(polinomio.fit_transform(x)), X_pol, Y_pol)
plt.title(f"Regresión Polinomial Grado {grado}")
plt.show()

accuracy_train = accuracy_score(Y_pol, y_pred)
print("Train Accuracy (Polynomial Logistic Regression):", accuracy_train)
