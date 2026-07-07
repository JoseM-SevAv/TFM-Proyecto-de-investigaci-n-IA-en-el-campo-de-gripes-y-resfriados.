# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:58:24 2026

@author: chema
"""

import os
import cv2
import numpy as np

# Creamos listas vacías que luego llenaremos con los datos de evaluación del modelo de Elastic Net.

X = []
y = []

# Creamos nuestras categorías de gargantas.

categorias = ["sano", "garganta inflamada", "garganta infectada"]

# Cargamos nuestras imágenes desde la carpeta del dataset, seleccionando las tres carpetas que se corresponderan con las 3 categorías.

for etiqueta, categoria in enumerate(categorias):

    carpeta = os.path.join("dataset/imagenes", categoria)

# Redimensionamos y escalamos las imágenes de la carpeta una por una.
  
    for archivo in os.listdir(carpeta):

        ruta = os.path.join(carpeta, archivo)

        img = cv2.imread(ruta)

        # Redimensionamos.
        img = cv2.resize(img, (64, 64))

        # Escalamos valores entre 0 y 1.
        img = img.astype(np.float32) / 255.0

        # Convertimos la imagen en un vector.
        caracteristicas = img.flatten()

        # Añadimos a las listas creadas antes las características de las imágenes extraídas y sus etiquetas.
      
        X.append(caracteristicas)
        y.append(etiqueta)

# Finalmente pasamos nuestras listas a formato array para que puedan ser usadas en la evaluación mediante el modelo Elastic Net.

X = np.array(X)
y = np.array(y)
