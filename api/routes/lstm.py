
"""Ruta del modelo LSTM nacional para predecir casos de dengue.

El LSTM prepara sus propias variables y genera las predicciones
de las próximas cuatro semanas.

La preparación de variables de la ANN se realiza por separado
desde el módulo correspondiente a la ANN.
"""

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.lstm_service import predecir_dengue


router = APIRouter(
    prefix="/lstm",
    tags=["LSTM - Predicción de dengue"]
)


class PrediccionRequest(BaseModel):
    fecha: date


@router.post("/predecir-dengue")
def ejecutar_prediccion(datos: PrediccionRequest):
    """Ejecuta el LSTM y devuelve sus predicciones nacionales."""

    try:
        resultado = predecir_dengue(datos.fecha)
        return resultado

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error del LSTM: {error}"
        ) from error
