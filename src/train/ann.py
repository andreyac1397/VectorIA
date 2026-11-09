
"""
VectorIA - Entrenamiento de ANN para recomendaciones preventivas.

División de entrenamiento y prueba:
- 80 % entrenamiento.
- 20 % prueba.
- GroupShuffleSplit por año y semana epidemiológica.
- Los cinco escenarios de una misma semana no se mezclan
  entre entrenamiento y prueba.

Ejecutar desde la raíz:
    python -m src.train.ann
"""

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR / "data" / "raw" / "vectoria_acciones.csv"
)

MODELS_DIR = BASE_DIR / "models" / "ANN"
RESULTS_DIR = BASE_DIR / "data" / "processed"

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


# ============================================================
# ENTRENAMIENTO
# ============================================================

def entrenar():

    # --------------------------------------------------------
    # 1. CARGAR DATASET
    # --------------------------------------------------------

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset: {DATASET_PATH}")
    print(f"Filas originales: {len(df)}")

    requeridas = FEATURES + [
        TARGET,
        "anio",
        "semana_epi"
    ]

    faltantes = [
        columna
        for columna in requeridas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )

    if df[requeridas].isna().any().any():
        raise ValueError(
            "Existen valores nulos en las columnas requeridas."
        )

    # --------------------------------------------------------
    # 2. VARIABLES DE ENTRADA
    # --------------------------------------------------------

    X = df[FEATURES].copy()

    for columna in FEATURES:
        X[columna] = pd.to_numeric(
            X[columna],
            errors="raise"
        )

    if not np.isfinite(
        X.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Hay valores no finitos en las variables."
        )

    if not X["aegypti_detectado"].isin([0, 1]).all():
        raise ValueError(
            "aegypti_detectado debe ser 0 o 1."
        )

    if not X["semana_epi"].between(1, 53).all():
        raise ValueError(
            "La semana epidemiológica debe estar entre 1 y 53."
        )

    # --------------------------------------------------------
    # 3. VARIABLE OBJETIVO
    # --------------------------------------------------------

    y = df[TARGET].astype(str).str.strip()

    if (y == "").any():
        raise ValueError(
            "Existen recomendaciones vacías."
        )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    if len(encoder.classes_) < 2:
        raise ValueError(
            "Se necesitan al menos dos clases."
        )

    # --------------------------------------------------------
    # 4. DIVISIÓN 80/20 AGRUPADA POR SEMANA
    # --------------------------------------------------------

    # Una semana se identifica por año + semana epidemiológica.
    # Por ejemplo, 2018-15 y 2024-15 son grupos distintos.

    anios = pd.to_numeric(
        df["anio"], errors="raise"
    ).astype(int)

    semanas = pd.to_numeric(
        df["semana_epi"], errors="raise"
    ).astype(int)

    grupos = (
        anios.astype(str)
        + "-"
        + semanas.astype(str).str.zfill(2)
    )

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED
    )

    train_idx, test_idx = next(
        gss.split(
            X,
            y_encoded,
            groups=grupos
        )
    )

    # Verificar que las semanas no estén mezcladas.

    grupos_train = set(grupos.iloc[train_idx])
    grupos_test = set(grupos.iloc[test_idx])

    assert grupos_train.isdisjoint(grupos_test), (
        "Error: hay semanas compartidas entre "
        "entrenamiento y prueba."
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    print("\n--- DIVISIÓN DE DATOS ---")

    print(
        f"Semanas de entrenamiento: {len(grupos_train)}"
    )

    print(
        f"Semanas de prueba: {len(grupos_test)}"
    )

    print(
        f"Registros de entrenamiento: {len(X_train)}"
    )

    print(
        f"Registros de prueba: {len(X_test)}"
    )

    print(
        "Semanas compartidas: ",
        len(grupos_train.intersection(grupos_test))
    )

    # --------------------------------------------------------
    # 5. ESCALAMIENTO
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # 6. SEMILLA
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # --------------------------------------------------------
    # 7. ARQUITECTURA ANN
    # --------------------------------------------------------

    model = Sequential([
        Input(shape=(len(FEATURES),)),

        Dense(128, activation="relu"),
        Dropout(0.2),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(
            len(encoder.classes_),
            activation="softmax"
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # 8. ENTRENAMIENTO
    # --------------------------------------------------------

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )

    # --------------------------------------------------------
    # 9. EVALUACIÓN
    # --------------------------------------------------------

    probabilidades = model.predict(
        X_test_scaled,
        verbose=0
    )

    y_pred = np.argmax(
        probabilidades,
        axis=1
    )

    etiquetas = np.arange(
        len(encoder.classes_)
    )

    reporte = classification_report(
        y_test,
        y_pred,
        labels=etiquetas,
        target_names=encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0
    )

    print(
        "\n--- REPORTE DE CLASIFICACIÓN (PRUEBA) ---"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=etiquetas,
            target_names=encoder.classes_.tolist(),
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nAccuracy de prueba: {accuracy:.4f}"
    )

    # --------------------------------------------------------
    # 10. GUARDAR MODELO Y ESCALADORES
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(MODEL_PATH)

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

    configuracion = {
        "features": FEATURES,
        "target": TARGET,
        "classes": encoder.classes_.tolist(),
        "seed": SEED,
        "test_size": 0.2,
        "split_method": "GroupShuffleSplit",
        "group_columns": ["anio", "semana_epi"],
        "dataset": "data/raw/vectoria_acciones.csv",
        "modelo": MODEL_PATH.name,
        "escalador": SCALER_PATH.name,
        "codificador": ENCODER_PATH.name,
    }

    FEATURES_PATH.write_text(
        json.dumps(
            configuracion,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 11. GUARDAR MÉTRICAS
    # --------------------------------------------------------

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

    pd.DataFrame(
        registros
    ).to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 12. GUARDAR HISTORIAL
    # --------------------------------------------------------

    historial = pd.DataFrame(
        history.history
    )

    historial.insert(
        0,
        "epoca",
        np.arange(1, len(historial) + 1)
    )

    historial.to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 13. MATRIZ DE CONFUSIÓN
    # --------------------------------------------------------

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=etiquetas
    )

    pd.DataFrame(
        matriz,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv(
        CONFUSION_PATH,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 14. RESULTADOS
    # --------------------------------------------------------

    print("\n--- ARCHIVOS GUARDADOS ---")

    archivos = [
        MODEL_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        FEATURES_PATH,
        METRICS_PATH,
        HISTORY_PATH,
        CONFUSION_PATH
    ]

    for archivo in archivos:
        print(
            archivo.relative_to(BASE_DIR)
        )

    print(
        "\nModelo original ann_model.keras conservado."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    entrenar()
