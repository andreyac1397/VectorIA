from pathlib import Path
from io import BytesIO
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


# ============================================================
# RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "cnn_mosquitos.keras"
)

CONFIG_PATH = (
    MODEL_DIR
    / "cnn_mosquitos_config.json"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

def cargar_configuracion():

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"No existe la configuración: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as archivo:

        return json.load(archivo)


CONFIG = cargar_configuracion()

CLASES = CONFIG["classes"]

IMG_WIDTH = int(
    CONFIG["input_width"]
)

IMG_HEIGHT = int(
    CONFIG["input_height"]
)


# ============================================================
# CARGA DEL MODELO
# ============================================================

_modelo = None


def cargar_modelo():
    """
    Carga el modelo una sola vez y lo mantiene
    disponible para las siguientes predicciones.
    """

    global _modelo

    if _modelo is None:

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"No existe el modelo: {MODEL_PATH}"
            )

        _modelo = keras.models.load_model(
            MODEL_PATH
        )

    return _modelo


# ============================================================
# APERTURA DE IMAGEN
# ============================================================

def _abrir_imagen(entrada):
    """
    Acepta:
    - ruta str
    - pathlib.Path
    - bytes
    - archivo tipo Streamlit UploadedFile
    - PIL.Image.Image
    """

    if isinstance(
        entrada,
        Image.Image
    ):
        return entrada.copy()

    if isinstance(
        entrada,
        (str, Path)
    ):

        ruta = Path(entrada)

        if not ruta.is_file():
            raise FileNotFoundError(
                f"No existe la imagen: {ruta}"
            )

        with Image.open(ruta) as imagen:
            return imagen.copy()

    if isinstance(
        entrada,
        bytes
    ):

        with Image.open(
            BytesIO(entrada)
        ) as imagen:
            return imagen.copy()

    # Compatible con Streamlit UploadedFile
    # y otros objetos tipo archivo.
    if hasattr(
        entrada,
        "read"
    ):

        if hasattr(
            entrada,
            "seek"
        ):
            entrada.seek(0)

        datos = entrada.read()

        with Image.open(
            BytesIO(datos)
        ) as imagen:
            return imagen.copy()

    raise TypeError(
        "La imagen debe ser una ruta, Path, bytes, "
        "PIL.Image o un objeto tipo archivo."
    )


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preparar_imagen(entrada):
    """
    Convierte una imagen al formato utilizado por la CNN.

    Salida:
        Tensor float32 de shape (1, 128, 128, 3).

    IMPORTANTE:
    No divide entre 255 porque cnn_mosquitos.keras
    contiene la capa Rescaling(1/255).
    """

    imagen = _abrir_imagen(
        entrada
    )

    imagen = imagen.convert(
        "RGB"
    )

    matriz = np.asarray(
        imagen,
        dtype=np.uint8
    )

    tensor = tf.convert_to_tensor(
        matriz,
        dtype=tf.uint8
    )

    tensor = tf.image.resize_with_pad(
        tensor,
        target_height=IMG_HEIGHT,
        target_width=IMG_WIDTH,
        method="bilinear"
    )

    tensor = tf.cast(
        tensor,
        tf.float32
    )

    tensor = tf.expand_dims(
        tensor,
        axis=0
    )

    return tensor


# ============================================================
# PREDICCIÓN
# ============================================================

def predecir_mosquito(entrada):
    """
    Clasifica una imagen en:

    - Aedes aegypti
    - Aedes albopictus
    - Culex

    Retorna un diccionario listo para consumir
    desde Streamlit, FastAPI u otro módulo Python.
    """

    modelo = cargar_modelo()

    tensor = preparar_imagen(
        entrada
    )

    probabilidades = modelo.predict(
        tensor,
        verbose=0
    )[0]

    indice = int(
        np.argmax(probabilidades)
    )

    probabilidades_dict = {
        clase: float(probabilidad)
        for clase, probabilidad
        in zip(
            CLASES,
            probabilidades
        )
    }

    resultado = {
        "class_id": indice,
        "especie": CLASES[indice],

        # Score softmax del modelo.
        # No implica probabilidad estadísticamente calibrada.
        "score": float(
            probabilidades[indice]
        ),

        "probabilidades": (
            probabilidades_dict
        ),

        "modelo": (
            CONFIG["model_name"]
        ),

        "input_size": [
            IMG_WIDTH,
            IMG_HEIGHT
        ]
    }

    return resultado


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":

    print(
        "Modelo:",
        MODEL_PATH
    )

    print(
        "Clases:",
        CLASES
    )

    print(
        "Input:",
        (
            IMG_HEIGHT,
            IMG_WIDTH,
            3
        )
    )