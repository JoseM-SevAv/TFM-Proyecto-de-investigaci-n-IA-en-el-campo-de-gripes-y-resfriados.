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

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from PIL import Image
import numpy as np

# --- Ejemplo de data augmentation para imagenes en HSV ---

image = Image.open("/content/calculo del entrenamiento de ia.png")

# Convertir a RGB si la imagen tiene 4 canales (RGBA)
if image.mode == 'RGBA':
    image = image.convert('RGB')

# Convertir la imagen PIL a un array de NumPy
image_np = np.array(image)

# Convertir el array de NumPy a un tensor de TensorFlow
image_tf = tf.convert_to_tensor(image_np)

# Asegurar que los valores de los píxeles estén entre 0 y 1 y en tipo float32
image_mod = tf.image.convert_image_dtype(image_tf, tf.float32)

# Generar una semilla (seed) aleatoria obligatoria para funciones sin estado (stateless)
seed = (1, 2)

# Aplicar las variaciones aleatorias HSV
image_mod = tf.image.stateless_random_hue(image_mod, max_delta=0.2, seed=seed)

# Cambia la saturación de forma aleatoria en el rango [0.5, 1.5]
image_mod = tf.image.stateless_random_saturation(image_mod, lower=0.5, upper=1.5, seed=seed)

# Cambia el brillo (Valor) de forma aleatoria en el rango [-0.2, 0.2]
image_mod = tf.image.stateless_random_brightness(image_mod, max_delta=0.2, seed=seed)

# Recortar los valores para que sigan dentro del rango válido [0.1]
image_mod = tf.clip_by_value(image_mod, 0.0, 1.0)

# Visualizar la imagen original y la imagen aumentada
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_tf.numpy())
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Augmented Image')
# Para mostrar con imshow, los valores deben estar en el rango [0, 1] para float o [0, 255] para int.
# La imagen aumentada ya está en [0, 1] debido a Rescaling(1./255).
plt.imshow(image_mod.numpy())
plt.axis('off')

plt.show()

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
# Convertir image_tf a float32 para Keras Sequential si aún no lo está (image_mod ya lo está, pero si queremos empezar de 'image_tf')
image_for_keras = tf.cast(image_tf, tf.float32)

# Aplicar redimensionamiento y reescalado
resized_rescaled_image = resize_and_rescale(image_for_keras) # Corrected: apply to image_for_keras (0-255 range)

# Aplicar la aumentación de datos (giro y volteo) a la imagen ya redimensionada y reescalada
augmented_image = data_augmentation(resized_rescaled_image)

# Visualizar la imagen original y la imagen aumentada
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_tf.numpy())
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Augmented Image')
# Para mostrar con imshow, los valores deben estar en el rango [0, 1] para float o [0, 255] para int.
# La imagen aumentada ya está en [0, 1] debido a Rescaling(1./255).
plt.imshow(augmented_image.numpy())
plt.axis('off')

plt.show()
