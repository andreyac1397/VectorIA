import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Cargar el dataset
data_path = os.path.join('..', '..', 'data', 'raw', 'vectoria_acciones.csv')
if not os.path.exists(data_path):
    data_path = 'vectoria_acciones.csv' # Ajustado según la ruta local

df = pd.read_csv(data_path)
print(f"Dataset cargado correctamente con {len(df)} filas.")

# Selección de variables del CSV
features = [
    'casos_pred_4sem', 'casos_umbral_endemico', 'lluvia_14d_mm',
    'temp_max', 'aegypti_detectado', 'semana_epi'
]
target = 'recomendacion'

X = df[features]
y = df[target]

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, shuffle=True
)

# Normalizar variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_classes = len(np.unique(y_encoded))

# Construir el modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluar el modelo
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


import os


os.makedirs(os.path.join('..', '..', 'models'), exist_ok=True)

# Guardar el modelo entrenado
model_path = os.path.join('..', '..', 'models', 'ann_model.keras')
model.save(model_path)
print(f"\nModelo guardado exitosamente en: {model_path}")