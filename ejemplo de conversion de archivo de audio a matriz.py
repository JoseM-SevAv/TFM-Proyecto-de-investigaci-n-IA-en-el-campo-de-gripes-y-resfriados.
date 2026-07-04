# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:26:08 2026

@author: chema
"""

import os
import librosa
import librosa.display
import numpy as np

# Cargamos el archivo de audio y guardamos en dos variables diferentes los valores 
# que componen la señal (variable: audio) y la frecuencia de muestreo (variable :sr).

audio, sr = librosa.load(librosa.ex("dataset/audios/tos.wav"), sr=None)

# Mediante la libreria librosa podemos extraer diferentes caracteristicas del sonido,
# para este proyecto usaremos MFCC (Mel Frequency Cepstral Coefficients), cuyo objetivo 
# es capturar el timbre de una señal de forma compacta, modelando cómo el oído humano 
# percibe las frecuencias, aquí extraemos 13 factores numericos (n_mfcc).

mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Pasamos los datos en una matriz para trabajar en elastic net

mfccs_matrix = np.matrix(mfccs)

# A partir de aquí podemos usar estos factores numericos para los siguentes pasos.

# La principal ventaja de mfcc es que condensa la información en pocos coeficientes, lo
# que nos viene bien para trabajar en el entrenamiento de modelos sin perder información esencial
