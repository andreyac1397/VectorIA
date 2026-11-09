"""Inferencia de MobileNetV2 para las tres especies de mosquitos de VectorIA.

El notebook de entrenamiento incluye preprocess_input DENTRO del modelo.
La imagen externa se envía RGB, 128x128, valores 0..255 (sin /255).
"""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_mosquitos.keras"
CLASES = ("Aedes aegypti", "Aedes albopictus", "Culex")
TAMANO = (128, 128)


@lru_cache(maxsize=1)
def cargar_mobilenet():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontró MobileNetV2: {MODEL_PATH}")
    modelo = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    if tuple(modelo.input_shape[1:]) != (128, 128, 3):
        raise ValueError(f"Entrada MobileNetV2 incompatible: {modelo.input_shape}")
    if int(modelo.output_shape[-1]) != len(CLASES):
        raise ValueError(f"Salida MobileNetV2 incompatible: {modelo.output_shape}")
    return modelo


def preparar_imagen(contenido: bytes):
    if not contenido:
        raise ValueError("La imagen está vacía.")
    try:
        with Image.open(BytesIO(contenido)) as imagen:
            imagen.verify()
        with Image.open(BytesIO(contenido)) as imagen:
            imagen = ImageOps.exif_transpose(imagen).convert("RGB")
            matriz = np.asarray(imagen, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("El archivo no es una imagen JPG o PNG válida.") from error
    tensor = tf.image.resize_with_pad(
        tf.convert_to_tensor(matriz, dtype=tf.uint8),
        target_height=TAMANO[0],
        target_width=TAMANO[1],
        method="bilinear",
    )
    # El notebook confirma que preprocess_input se ejecuta dentro del .keras.
    return tf.expand_dims(tf.cast(tensor, tf.float32), axis=0)


def predecir_mobilenet(contenido: bytes) -> dict:
    imagen = preparar_imagen(contenido)
    modelo = cargar_mobilenet()
    probabilidades = np.asarray(modelo.predict(imagen, verbose=0)[0], dtype=float)
    if (probabilidades.shape != (len(CLASES),)
            or not np.isfinite(probabilidades).all()
            or np.any(probabilidades < 0)):
        raise RuntimeError("MobileNetV2 devolvió valores de salida inválidos.")
    indice = int(np.argmax(probabilidades))
    scores = {clase: round(float(valor), 6)
              for clase, valor in zip(CLASES, probabilidades)}
    especie = CLASES[indice]
    return {
        "estado": "correcto",
        "modelo": "MobileNetV2",
        "modelo_utilizado": MODEL_PATH.name,
        "class_id": indice,
        "especie": especie,
        "score": round(float(probabilidades[indice]), 6),
        "confianza": round(float(probabilidades[indice]), 6),
        "probabilidades": scores,
        "input_size": list(TAMANO),
        # La ANN entrena esta variable para Aedes aegypti, no para todo el género Aedes.
        "aegypti_detectado": int(especie == "Aedes aegypti"),
        "advertencia": "Clasificación experimental: el valor Softmax no es una probabilidad calibrada.",
    }
