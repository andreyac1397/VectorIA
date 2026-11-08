from fastapi import APIRouter
from pydantic import BaseModel
from datetime import date

router = APIRouter(
    prefix="/lstm",
    tags=["LSTM - Predicción de dengue"]
)


# Datos que recibirá la API
class PrediccionRequest(BaseModel):
    fecha: date


# Endpoint de predicción
@router.post("/predecir-dengue")
def predecir_dengue(datos: PrediccionRequest):

    # El modelo LSTM se integrará posteriormente
    return {
        "estado": "pendiente",
        "mensaje": "El modelo LSTM todavía no está conectado.",
        "fecha_seleccionada": datos.fecha,
        "semanas_prediccion": 4
    }