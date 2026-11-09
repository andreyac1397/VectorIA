"""Inferencia de la ANN v2, con los artefactos exactos de entrenamiento."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "ANN"
MODEL_PATH = MODELS_DIR / "ann_model_v2.keras"
SCALER_PATH = MODELS_DIR / "ann_scaler_v2.joblib"
ENCODER_PATH = MODELS_DIR / "ann_label_encoder_v2.joblib"
FEATURES_PATH = MODELS_DIR / "ann_features_v2.json"

FEATURES_ESPERADAS = [
    "casos_pred_4sem",
    "casos_umbral_endemico",
    "lluvia_14d_mm",
    "temp_max",
    "aegypti_detectado",
    "semana_epi",
]


@lru_cache(maxsize=1)
def cargar_recursos_ann():
    for ruta in (MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta el archivo de la ANN: {ruta}")

    configuracion = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    features = configuracion.get("features")
    if features != FEATURES_ESPERADAS:
        raise ValueError(
            f"El orden de variables no coincide con el entrenamiento: {features}"
        )

    modelo = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    if getattr(scaler, "n_features_in_", None) != len(features):
        raise ValueError("El escalador de la ANN no coincide con sus seis variables")
    if modelo.input_shape[-1] != len(features):
        raise ValueError("El modelo ANN no recibe las seis variables esperadas")
    if modelo.output_shape[-1] != len(encoder.classes_):
        raise ValueError("Las clases del modelo y del codificador no coinciden")
    if configuracion.get("classes") != encoder.classes_.tolist():
        raise ValueError("El orden de las clases en el JSON y el codificador no coincide")

    return modelo, scaler, encoder, features


def recomendar_accion(
    casos_pred_4sem: float,
    umbral_endemico: float,
    lluvia_14d_mm: float,
    temp_max: float,
    aegypti_det: int,
    semana_epi: int,
):
    """Procesa entradas sin reentrenar ni reemplazar los artefactos v2."""
    modelo, scaler, encoder, features = cargar_recursos_ann()
    valores = {
        "casos_pred_4sem": casos_pred_4sem,
        "casos_umbral_endemico": umbral_endemico,
        "lluvia_14d_mm": lluvia_14d_mm,
        "temp_max": temp_max,
        "aegypti_detectado": aegypti_det,
        "semana_epi": semana_epi,
    }
    entrada = pd.DataFrame([valores], columns=features)
    normalizado = scaler.transform(entrada).astype(np.float32)
    probabilidades = np.asarray(modelo.predict(normalizado, verbose=0)[0], dtype=float)
    if len(probabilidades) != len(encoder.classes_) or not np.isfinite(probabilidades).all():
        raise RuntimeError("El modelo ANN no produjo probabilidades válidas")

    indice = int(np.argmax(probabilidades))
    recomendacion = str(encoder.inverse_transform([indice])[0])
    conf = float(probabilidades[indice])

    return {
        "estado": "correcto",
        "modelo": "ANN",
        "modelo_utilizado": MODEL_PATH.name,
        "recomendacion": recomendacion,
        "confianza": round(conf, 6),
        "probabilidades": {
            str(clase): round(float(p), 6)
            for clase, p in zip(encoder.classes_, probabilidades)
        },
        "semana_epi": semana_epi,
        "datos_recibidos": {
            "casos_pred_4sem": casos_pred_4sem,
            "umbral_endemico": umbral_endemico,
            "lluvia_14d_mm": lluvia_14d_mm,
            "temp_max": temp_max,
            "aegypti_det": aegypti_det,
        },
        "explicacion": (
            "Clasificación generada por la ANN entrenada con vectoria_acciones.csv. "
            "Es una recomendación experimental; no sustituye criterios sanitarios."
        ),
    }
