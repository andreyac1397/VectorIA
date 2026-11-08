from pathlib import Path

# Ruta donde se guardará el modelo entrenado
BASE_DIR = Path(__file__).resolve().parents[2]
MODELO_PATH = BASE_DIR / "models" / "ann_model.keras"


def recomendar_accion(
    casos_pred_4sem: float,
    umbral_endemico: float,
    lluvia_14d_mm: float,
    temp_max: float,
    aegypti_det: int
):
    """
    Recibe los datos de dengue, clima y detección
    del mosquito para recomendar una acción preventiva.
    """

    # Comprobar si existe el modelo
    if not MODELO_PATH.exists():
        return {
            "estado": "pendiente",
            "mensaje": "El modelo ANN todavía no está disponible."
        }

    # Pendiente:
    # 1. Cargar el modelo ANN entrenado.
    # 2. Preparar las variables de entrada.
    # 3. Aplicar el preprocesamiento del entrenamiento.
    # 4. Ejecutar la predicción.
    # 5. Interpretar la clasificación obtenida.
    # 6. Devolver la recomendación preventiva.

    return {
        "estado": "pendiente",
        "mensaje": "La integración del modelo ANN está pendiente."
    }