# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:35:02 2026

@author: chema
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split

# 1. Supongamos que X_features es tu matriz de características extraídas (imágenes o audio)
# y 'y' es tu variable objetivo.
# Usando el código anterior podemos meter nuestras matrices en variables

X_features = mfccs_matrix
X_features_2 = imagen_matrix

# 2. Estandarizar los datos

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Dividir en datos de entrenamiento y prueba, usaremos 20% de test_size

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Entrenar Elastic Net (con validación cruzada para ajustar hiperparámetros)
# l1_ratio = 1.0 es Lasso puro, l1_ratio = 0.0 es Ridge puro, por lo que usaremos
# diferentes l1_ratio entre 0 y 1 para ajustar elastic net.

elastic_net = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv=5, random_state=42)
elastic_net.fit(X_train, y_train)

# Resultados

print(f"Mejor alpha: {elastic_net.alpha_}")
print(f"Mejor l1_ratio: {elastic_net.l1_ratio_}")
print(f"Score en test: {elastic_net.score(X_test, y_test)}")