from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(
    prefix="/ann",
    tags=["ANN - Recomendaciones preventivas"]
)


# Datos que recibirá la API
class RecomendacionRequest(BaseModel):
    casos_pred_4sem: float
    umbral_endemico: float
    lluvia_14d_mm: float
    temp_max: float
    aegypti_det: int


# Endpoint de recomendación
@router.post("/recomendar-accion")
def recomendar_accion(datos: RecomendacionRequest):

    # El modelo ANN se integrará posteriormente
    return {
        "estado": "pendiente",
        "mensaje": "El modelo ANN todavía no está conectado.",
        "datos_recibidos": datos.model_dump()
    }