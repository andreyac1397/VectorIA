
"""Rutas FastAPI para recomendaciones preventivas de la ANN v2.

Permite:
1. Preparar las variables históricas de ANN sin ejecutar el modelo.
2. Generar una recomendación utilizando las variables preparadas.

El LSTM funciona de manera independiente.
"""

from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.services.ann_service import (
    recomendar_accion as predecir_ann,
)

from api.services.contexto_ann_service import (
    preparar_contexto_ann,
)

from src.train.lstm import anio_semana_epi


router = APIRouter(
    prefix="/ann",
    tags=["ANN - Recomendaciones preventivas"],
)


# ============================================================
# 1. MODELOS DE ENTRADA
# ============================================================

class PrepararVariablesRequest(BaseModel):
    """Datos que necesita ANN para preparar sus variables."""

    casos_pred_4sem: int = Field(
        ge=0,
        description="Total de casos pronosticados por el LSTM.",
    )

    fecha_referencia: date = Field(
        description="Fecha de la primera semana pronosticada.",
    )


class RecomendacionRequest(PrepararVariablesRequest):
    """Datos para ejecutar la recomendación de ANN.

    Permite dos formas:
    - Enviar casos, fecha y Aedes para preparar automáticamente.
    - Enviar también las cuatro variables preparadas para
      ejecutar directamente la ANN.
    """

    aegypti_det: int = Field(
        ge=0,
        le=1,
        description="0 = No detectado, 1 = Detectado.",
    )

    umbral_endemico: float | None = Field(
        default=None,
        ge=0,
    )

    lluvia_14d_mm: float | None = Field(
        default=None,
        ge=0,
    )

    temp_max: float | None = Field(
        default=None,
        ge=-20,
        le=60,
    )

    semana_epi: int | None = Field(
        default=None,
        ge=1,
        le=53,
    )

    @model_validator(mode="after")
    def validar_variables_preparadas(self):
        variables = (
            self.umbral_endemico,
            self.lluvia_14d_mm,
            self.temp_max,
            self.semana_epi,
        )

        cantidad = sum(
            valor is not None
            for valor in variables
        )

        if cantidad not in (0, 4):
            raise ValueError(
                "Envíe las cuatro variables preparadas "
                "o ninguna de ellas."
            )

        if cantidad == 4:
            _, semana_calculada = anio_semana_epi(
                self.fecha_referencia
            )

            if self.semana_epi != semana_calculada:
                raise ValueError(
                    "semana_epi no coincide con "
                    "fecha_referencia."
                )

        return self


# ============================================================
# 2. FUNCIÓN AUXILIAR
# ============================================================

def obtener_contexto(casos_pred_4sem, fecha_referencia):
    """Prepara las variables históricas y maneja sus errores."""

    try:
        return preparar_contexto_ann(
            casos_pred_4sem=casos_pred_4sem,
            fecha_referencia=fecha_referencia,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Error preparando variables ANN: {error}",
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Falta un dataset necesario: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible preparar las variables ANN: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error


# ============================================================
# 3. ENDPOINT: PREPARAR VARIABLES
# ============================================================

@router.post("/preparar-variables")
def preparar_variables(datos: PrepararVariablesRequest):
    """Prepara las variables de ANN sin ejecutar el modelo."""

    contexto = obtener_contexto(
        casos_pred_4sem=datos.casos_pred_4sem,
        fecha_referencia=datos.fecha_referencia,
    )

    return {
        "estado": "correcto",
        "modelo": "ANN",
        "accion": "preparacion_variables",
        "contexto_ann": contexto,
    }


# ============================================================
# 4. ENDPOINT: GENERAR RECOMENDACIÓN
# ============================================================

@router.post("/recomendar-accion")
def recomendar_accion(datos: RecomendacionRequest):
    """Ejecuta la ANN utilizando sus seis variables."""

    variables_completas = all(
        valor is not None
        for valor in (
            datos.umbral_endemico,
            datos.lluvia_14d_mm,
            datos.temp_max,
            datos.semana_epi,
        )
    )

    if variables_completas:
        # Las variables ya fueron preparadas.
        # No es necesario volver a consultar los datasets.
        contexto = {
            "casos_pred_4sem": datos.casos_pred_4sem,
            "umbral_endemico": datos.umbral_endemico,
            "lluvia_14d_mm": datos.lluvia_14d_mm,
            "temp_max": datos.temp_max,
            "semana_epi": datos.semana_epi,
            "fecha_referencia": (
                datos.fecha_referencia.isoformat()
            ),
            "origen_clima": "variables_proporcionadas",
            "origen_umbral": "variables_proporcionadas",
            "advertencia": (
                "La ANN utilizó las variables suministradas. "
                "No se recalcularon los valores históricos "
                "durante esta solicitud."
            ),
        }

    else:
        # Compatibilidad con las solicitudes anteriores.
        # Prepara automáticamente las variables si no se envían.
        contexto = obtener_contexto(
            casos_pred_4sem=datos.casos_pred_4sem,
            fecha_referencia=datos.fecha_referencia,
        )

    # Ejecutar la inferencia de ANN.
    try:
        resultado = predecir_ann(
            casos_pred_4sem=contexto["casos_pred_4sem"],
            umbral_endemico=contexto["umbral_endemico"],
            lluvia_14d_mm=contexto["lluvia_14d_mm"],
            temp_max=contexto["temp_max"],
            aegypti_det=datos.aegypti_det,
            semana_epi=contexto["semana_epi"],
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Falta un archivo del modelo ANN: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible ejecutar la ANN: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error

    # Conservar la estructura de respuesta utilizada por Streamlit.
    resultado["fecha_referencia"] = (
        datos.fecha_referencia.isoformat()
    )

    resultado["contexto_ann"] = contexto

    resultado["origen_clima"] = contexto["origen_clima"]

    resultado["origen_umbral"] = contexto["origen_umbral"]

    resultado["advertencia_datos"] = contexto["advertencia"]

    return resultado
