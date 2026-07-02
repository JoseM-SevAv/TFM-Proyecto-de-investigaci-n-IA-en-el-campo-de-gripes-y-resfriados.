# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:44:47 2026

@author: chema
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from PIL import Image

# --- Ejemplo de data augmentation para imagenes en HSV ---

image = Image.open("data/image.jpg")

# Asegurar que los valores de los píxeles estén entre 0 y 1

image_mod = tf.image.convert_image_dtype(image, tf.float32)
    
# Generar una semilla (seed) aleatoria obligatoria para funciones sin estado (stateless)

seed = (1, 2)
    
# Aplicar las variaciones aleatorias HSV

image_mod = tf.image.stateless_random_hue(image, max_delta=0.2, seed=seed)
    
# Cambia la saturación de forma aleatoria en el rango [0.5, 1.5]

image_mod = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed)
    
# Cambia el brillo (Valor) de forma aleatoria en el rango [-0.2, 0.2]

image_mod = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
    
# Recortar los valores para que sigan dentro del rango válido [0.1]

image = tf.clip_by_value(image, 0.0, 1.0)

# --- Ejemplo de data augmentation para rescalar imagenes ---

IMG_SIZE = 180

# Con la función Sequential de keras rescalamos las imagenes

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

# También podemos girar las imagenes de forma aleatoria

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])