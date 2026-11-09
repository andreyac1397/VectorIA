from pathlib import Path

# Ruta donde se guardará el modelo entrenado
BASE_DIR = Path(__file__).resolve().parents[2]
MODELO_PATH = BASE_DIR / "models" / "cnn_model.keras"


def identificar_mosquito(contenido_imagen: bytes):
    """
    Recibe una imagen y devuelve la clasificación del mosquito.
    """

    # Comprobar si existe el modelo
    if not MODELO_PATH.exists():
        return {
            "estado": "pendiente",
            "mensaje": "El modelo CNN todavía no está disponible."
        }

    # Pendiente:
    # 1. Cargar el modelo CNN entrenado.
    # 2. Decodificar y preprocesar la imagen.
    # 3. Ejecutar la predicción.
    # 4. Obtener la especie y su probabilidad.

    return {
        "estado": "pendiente",
        "mensaje": "La integración del modelo CNN está pendiente."
    }