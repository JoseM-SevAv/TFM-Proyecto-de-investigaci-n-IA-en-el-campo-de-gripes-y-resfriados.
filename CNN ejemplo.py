# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:14:32 2026

@author: chema
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Tenemos dos carpetas para imagenes y test porque ImageDataGenerator solo permite 
# La separación en dos carpetas (train y validation)

BASE_PATH = '/data/content/imagenes'
TEST_PATH = 'data/content/imagenes_test'

# Generador train + validación (60%/20%)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

# El batch_size indica el calculo de la cantidad de datos que pasan a la vez, se modificará
# en función a las necesidades del modelo para evitar el sobreajuste
# Para datos que tienen mas de 2 tipos de variables indicaremos class_mode='categorical'

train_gen = datagen.flow_from_directory(
    BASE_PATH, target_size=(100,100),
    batch_size=32, class_mode='categorical', subset='training', seed=42
)
val_gen = datagen.flow_from_directory(
    BASE_PATH, target_size=(100,100),
    batch_size=32, class_mode='categorical', subset='validation', seed=42
)

# Generador test separado (20% restante)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_PATH, target_size=(100,100),
    batch_size=32, class_mode='categorical', shuffle=False)

model = models.Sequential([ # Marcamos cada una de las diferentes capas.
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)), # Función de activación relu
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='softmax')  # Clasificación multiclase en la ultima capa
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) # Optimizador adam, famoso por su estabilidad y rapidez, 
# función de perdida categorical_crossentropy para datos multiclase
model.summary()

# Corremos el modelo indicado durante 10 epocas

history = model.fit(train_gen, epochs=10, validation_data=val_gen)

# Ploteamos la precisión y la función de perdida

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Validación')
ax1.set_title('Función de Pérdida'); ax1.legend()

ax2.plot(history.history['accuracy'], label='Train')
ax2.plot(history.history['val_accuracy'], label='Validación')
ax2.set_title('Precisión'); ax2.legend()

plt.tight_layout()
plt.show()

# Predicciones

y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()
y_true = test_gen.classes

# Matriz de confusión

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Parasitized','Uninfected']).plot()
plt.title('Matriz de Confusión')
plt.show()

# Métricas

print(classification_report(y_true, y_pred,
      target_names=['Uninfected','Parasitized']))
