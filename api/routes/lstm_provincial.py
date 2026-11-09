"""Ruta opcional del bonus LSTM provincial; no afecta /lstm/predecir-dengue."""
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services.lstm_provincial_service import predecir_dengue_provincial

router = APIRouter(prefix='/lstm-provincial', tags=['Bonus - LSTM provincial'])


class PrediccionProvincialRequest(BaseModel):
    fecha: date = Field(description='Fecha de 2026; se usa el domingo coincidente o posterior')
    provincia: str | None = Field(default=None, description='Provincia o null para todas (mapa)')


@router.post('/predecir-dengue')
def predecir_dengue(datos: PrediccionProvincialRequest):
    try:
        return predecir_dengue_provincial(fecha=datos.fecha, provincia=datos.provincia)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f'Falta archivo del modelo provincial: {exc}') from exc
    except Exception as exc:
        # No afecta otros endpoints; mensaje suficiente para depuración.
        raise HTTPException(status_code=500, detail=f'No se pudo generar la predicción provincial: {exc}') from exc
