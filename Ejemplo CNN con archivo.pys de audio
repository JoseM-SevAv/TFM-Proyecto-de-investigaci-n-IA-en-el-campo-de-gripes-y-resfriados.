import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
import librosa

# Reusamos el ejemplo de conversión de archivos de audio para este entrenamiento.

# Separamos el dataset, un 60% del dataset será utilizado para entrenamiento, el 40% restante será usado para testear y validar.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Del 40% restante dividimos un 20% para testeo y el 20% restante para validación.

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Creamos el modelo de capas.

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)), 
    layers.Dense(64, activation='relu'), # Función de activación "relu"
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(np.unique(y)), activation='softmax')  # Clasificación multiclase "softmax" en la última capa para 3 categorías.
])

model.compile(optimizer='adam', # Optimizador "adam", famoso por su estabilidad y rapidez, 
              loss='categorical_crossentropy', # función de pérdida categorical_crossentropy para datos multiclase
              metrics=['accuracy'])
model.summary()

# Pasamos las variables "y" a categórica para aplicar el modelo de entrenamiento al dataset

num_classes = len(np.unique(y))
y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val_ohe = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Entrenamos el modelo con 10 épocas.

history = model.fit(X_train, y_train_ohe, epochs=10, validation_data=(X_val, y_val_ohe))

# Ploteamos los resultados, la función de pérdida y la precisión del modelo.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

ax1.plot(history.history['loss'], label='Train')
ax1.plot(history.history['val_loss'], label='Validación')
ax1.set_title('Función de Pérdida'); ax1.legend()

ax2.plot(history.history['accuracy'], label='Train')
ax2.plot(history.history['val_accuracy'], label='Validación')
ax2.set_title('Precisión'); ax2.legend()

plt.tight_layout()
plt.show()

# Predicciones.

y_pred_probs = model.predict(X_train)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_train # Use the original integer labels

# Obtener las etiquetas de clase en el orden correcto
# 'categorias' list already contains the ordered class names
ordered_class_names = categorias

# Matriz de confusión.

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=ordered_class_names).plot()
plt.title('Matriz de Confusión')
plt.show()

# Métricas.

print(classification_report(y_true, y_pred,
      target_names=ordered_class_names))
