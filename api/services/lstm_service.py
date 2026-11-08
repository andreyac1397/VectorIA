from pathlib import Path
from datetime import date

# Ruta donde se guardará el modelo entrenado
BASE_DIR = Path(__file__).resolve().parents[2]
MODELO_PATH = BASE_DIR / "models" / "lstm_model.keras"


def predecir_dengue(fecha: date):
    """
    Recibe una fecha y solicita la predicción
    de casos de dengue para las próximas 4 semanas.
    """

    # Comprobar si existe el modelo
    if not MODELO_PATH.exists():
        return {
            "estado": "pendiente",
            "mensaje": "El modelo LSTM todavía no está disponible."
        }

    # Pendiente:
    # 1. Cargar el modelo LSTM entrenado.
    # 2. Consultar los datos históricos de dengue y clima.
    # 3. Preparar las ventanas de 26 semanas.
    # 4. Aplicar el escalado utilizado en el entrenamiento.
    # 5. Ejecutar la predicción de las próximas 4 semanas.
    # 6. Devolver las fechas y los casos pronosticados.

    return {
        "estado": "pendiente",
        "mensaje": "La integración del modelo LSTM está pendiente.",
        "fecha_seleccionada": fecha.isoformat(),
        "semanas_prediccion": 4
    }