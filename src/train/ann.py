"""Entrenamiento reproducible de la ANN de recomendaciones preventivas de VectorIA.

Ejecutar desde cualquier directorio:
    python -m src.train.ann

Los nuevos artefactos se guardan con sufijo _v2 para no sobrescribir
el modelo original antes de verificar resultados.
"""
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "data" / "raw" / "vectoria_acciones.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "data" / "processed"

# Guardamos una versión nueva para preservar ann_model.keras de la compañera.
MODEL_PATH = MODELS_DIR / "ann_model_v2.keras"
SCALER_PATH = MODELS_DIR / "ann_scaler_v2.joblib"
ENCODER_PATH = MODELS_DIR / "ann_label_encoder_v2.joblib"
FEATURES_PATH = MODELS_DIR / "ann_features_v2.json"
METRICS_PATH = RESULTS_DIR / "metricas_ann_v2.csv"
HISTORY_PATH = RESULTS_DIR / "historial_entrenamiento_ann_v2.csv"
CONFUSION_PATH = RESULTS_DIR / "matriz_confusion_ann_v2.csv"

FEATURES = [
    "casos_pred_4sem",
    "casos_umbral_endemico",
    "lluvia_14d_mm",
    "temp_max",
    "aegypti_detectado",
    "semana_epi",
]
TARGET = "recomendacion"
SEED = 42
EPOCHS = 50
BATCH_SIZE = 32


def entrenar():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATASET_PATH}\n"
            "Coloque vectoria_acciones.csv en data/raw/."
        )

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Filas originales: {len(df)}")

    requeridas = FEATURES + [TARGET]
    faltantes = [col for col in requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el CSV: {faltantes}")

    # Validaciones sin alterar ni eliminar filas silenciosamente.
    if df[requeridas].isna().any().any():
        nulos = df[requeridas].isna().sum()
        raise ValueError(f"Hay valores nulos en los datos utilizados:\n{nulos[nulos > 0]}")

    X = df[FEATURES].copy()
    for columna in FEATURES:
        X[columna] = pd.to_numeric(X[columna], errors="raise")
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError("Hay valores infinitos o no finitos en las variables de entrada.")

    y = df[TARGET].astype(str).str.strip()
    if (y == "").any():
        raise ValueError("Hay recomendaciones vacías.")

    if not X["aegypti_detectado"].isin([0, 1]).all():
        raise ValueError("aegypti_detectado debe contener únicamente 0 y 1.")
    if not X["semana_epi"].between(1, 53).all():
        raise ValueError("semana_epi debe estar entre 1 y 53.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    if len(encoder.classes_) < 2:
        raise ValueError("Se necesitan al menos dos clases de recomendación.")

    # División 80/20 agrupando por semana epidemiológica (semana_epi).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=df["semana_epi"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # IMPORTANTE: ajustar solo con entrenamiento y reutilizar en pruebas y producción.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(len(FEATURES),)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    # Evaluación del conjunto de prueba, que no participó en el ajuste.
    probabilidades = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(probabilidades, axis=1)
    etiquetas = np.arange(len(encoder.classes_))
    reporte = classification_report(
        y_test,
        y_pred,
        labels=etiquetas,
        target_names=encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    print("\n--- REPORTE DE CLASIFICACIÓN (PRUEBA) ---")
    print(classification_report(
        y_test, y_pred, labels=etiquetas,
        target_names=encoder.classes_.tolist(), zero_division=0,
    ))
    print(f"Accuracy de prueba: {accuracy_score(y_test, y_pred):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    FEATURES_PATH.write_text(
        json.dumps({
            "features": FEATURES,
            "target": TARGET,
            "classes": encoder.classes_.tolist(),
            "seed": SEED,
            "test_size": 0.2,
            "dataset": "data/raw/vectoria_acciones.csv",
            "modelo": MODEL_PATH.name,
            "escalador": SCALER_PATH.name,
            "codificador": ENCODER_PATH.name,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registros = []
    for nombre, datos in reporte.items():
        if isinstance(datos, dict):
            registros.append({
                "clase": nombre,
                "precision": datos.get("precision"),
                "recall": datos.get("recall"),
                "f1_score": datos.get("f1-score"),
                "support": datos.get("support"),
                "accuracy_global": reporte["accuracy"],
            })
    pd.DataFrame(registros).to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    historial = pd.DataFrame(history.history)
    historial.insert(0, "epoca", np.arange(1, len(historial) + 1))
    historial.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

    matriz = confusion_matrix(y_test, y_pred, labels=etiquetas)
    pd.DataFrame(
        matriz, index=encoder.classes_, columns=encoder.classes_
    ).to_csv(CONFUSION_PATH, encoding="utf-8-sig")

    print("\n--- ARCHIVOS GUARDADOS ---")
    for archivo in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH,
                    METRICS_PATH, HISTORY_PATH, CONFUSION_PATH]:
        print(archivo.relative_to(BASE_DIR))
    print("\nModelo original ann_model.keras conservado sin cambios.")


if __name__ == "__main__":
    entrenar()