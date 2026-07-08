# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:26:08 2026

@author: chema
"""

import os
import librosa
import numpy as np

# Mediante la libreria librosa podemos extraer diferentes caracteristicas del sonido,
# para este proyecto usaremos MFCC (Mel Frequency Cepstral Coefficients), cuyo objetivo 
# es capturar el timbre de una señal de forma compacta, modelando cómo el oído humano 
# percibe las frecuencias, aquí extraemos 20 factores numericos (n_mfcc).

# La principal ventaja de mfcc es que condensa la información en pocos coeficientes, lo
# que nos viene bien para trabajar en el entrenamiento de modelos sin perder información esencial

# Creamos listas vacías que luego llenaremos con los datos de evaluación del modelo de Elastic Net.

X = []
y = []

# Creamos nuestras categorías de tos.

categorias = ["sano", "tos seca", "tos productiva"]

# Cargamos nuestros audios desde la carpeta del dataset, seleccionando las tres carpetas que se corresponderan con las 3 categorías.

for etiqueta, categoria in enumerate(categorias):

    carpeta = os.path.join("dataset/imagenes", categoria)

# Cargamos los audios de las carpetas una por una.
  
    for archivo in os.listdir(carpeta):

        ruta = os.path.join(carpeta, archivo)

        # Cargamos los audios
      
        audio, sr = librosa.load(ruta, sr=22050)

        # Extraemos 20 MFCC
      
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20
        )

        # Calculamos el promedio temporal de cada coeficiente
        caracteristicas = np.mean(mfcc, axis=1)

        # Metemos las características y etiquetas obtenidas en las listas y de ahí a los arrays.

        X.append(caracteristicas)
        y.append(etiqueta)

X = np.array(X)
y = np.array(y)
