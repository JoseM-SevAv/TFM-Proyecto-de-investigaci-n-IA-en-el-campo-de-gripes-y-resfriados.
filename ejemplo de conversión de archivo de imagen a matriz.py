# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:58:24 2026

@author: chema
"""

import os
import cv2
import numpy as np

# Cargar la imagen

imagen = cv2.imread('dataset/imagenes/garganta.jpg')

# Convertir a matriz numérica, en este caso, y a pesar de usar np.array se nos mostrará una estructura más acorde con una matriz que con un array.

imagen_matrix = np.array(imagen)

# A partir de aquí, al igual que con los datos de sonido, podemos usar estos datos
# Para un modelo de elastic net.
