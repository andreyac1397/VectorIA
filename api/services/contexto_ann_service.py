
"""Prepara automáticamente las variables históricas de la ANN.

La ANN recibe:
- Casos pronosticados por el LSTM para cuatro semanas.
- Fecha de referencia de la primera semana pronosticada.

Este servicio obtiene:
- Umbral endémico.
- Lluvia acumulada de 14 días.
- Temperatura máxima.
- Semana epidemiológica.

Fuentes:
- vectoria_acciones.csv: referencias históricas de ANN.
- vectoria_dengue_nacional_real.csv: cálculo de umbrales futuros.

La detección de Aedes se recibe por separado desde la
interfaz y posteriormente desde la CNN.

Las referencias para fechas futuras son estimaciones
históricas y no observaciones meteorológicas.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.train.lstm import anio_semana_epi


BASE_DIR = Path(__file__).resolve().parents[2]

ACCIONES_PATH = (
    BASE_DIR / "data" / "raw" / "vectoria_acciones.csv"
)

NACIONAL_PATH = (
    BASE_DIR / "data" / "raw" / "vectoria_dengue_nacional_real.csv"
)


@lru_cache(maxsize=1)
def cargar_datos_ann():
    """Carga y valida las referencias históricas del dataset ANN."""

    if not ACCIONES_PATH.is_file():
        raise FileNotFoundError(
            f"No existe el dataset ANN: {ACCIONES_PATH}"
        )

    datos = pd.read_csv(ACCIONES_PATH)

    requeridas = [
        "fecha_inicio_semana",
        "semana_epi",
        "casos_umbral_endemico",
        "lluvia_14d_mm",
        "temp_max",
    ]

    faltantes = [
        columna
        for columna in requeridas
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas en vectoria_acciones.csv: {faltantes}"
        )

    datos = datos.copy()

    datos["fecha_inicio_semana"] = pd.to_datetime(
        datos["fecha_inicio_semana"],
        errors="raise",
    ).dt.normalize()

    columnas_numericas = [
        "semana_epi",
        "casos_umbral_endemico",
        "lluvia_14d_mm",
        "temp_max",
    ]

    for columna in columnas_numericas:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="raise",
        )

    if datos[requeridas].isna().any().any():
        raise ValueError(
            "El dataset ANN contiene valores nulos "
            "en las columnas necesarias."
        )

    if not np.isfinite(
        datos[columnas_numericas].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "El dataset ANN contiene valores no finitos."
        )

    if not datos["semana_epi"].between(1, 53).all():
        raise ValueError(
            "Existen semanas epidemiológicas inválidas."
        )

    if (
        datos["lluvia_14d_mm"].lt(0).any()
        or datos["casos_umbral_endemico"].lt(0).any()
    ):
        raise ValueError(
            "La lluvia y el umbral no pueden ser negativos."
        )

    # Cada semana contiene cinco escenarios.
    # Se consolida una sola referencia por semana.
    referencias = (
        datos.groupby("fecha_inicio_semana", as_index=False)
        .agg(
            semana_epi=("semana_epi", "first"),
            casos_umbral_endemico=(
                "casos_umbral_endemico", "median"
            ),
            lluvia_14d_mm=("lluvia_14d_mm", "median"),
            temp_max=("temp_max", "median"),
        )
        .sort_values("fecha_inicio_semana")
        .reset_index(drop=True)
    )

    referencias["anio"] = (
        referencias["fecha_inicio_semana"].dt.year
    )

    return referencias


@lru_cache(maxsize=1)
def cargar_casos_nacionales():
    """Carga los casos históricos reales del dataset nacional."""

    if not NACIONAL_PATH.is_file():
        raise FileNotFoundError(
            f"No existe el dataset nacional: {NACIONAL_PATH}"
        )

    datos = pd.read_csv(NACIONAL_PATH)

    requeridas = [
        "fecha_inicio_semana",
        "casos_dengue",
    ]

    faltantes = [
        columna
        for columna in requeridas
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas en el dataset nacional: {faltantes}"
        )

    datos = datos[requeridas].copy()

    datos["fecha_inicio_semana"] = pd.to_datetime(
        datos["fecha_inicio_semana"],
        errors="raise",
    ).dt.normalize()

    datos["casos_dengue"] = pd.to_numeric(
        datos["casos_dengue"],
        errors="raise",
    )

    if datos.isna().any().any():
        raise ValueError(
            "El dataset nacional contiene valores nulos."
        )

    if (
        not np.isfinite(datos["casos_dengue"]).all()
        or datos["casos_dengue"].lt(0).any()
    ):
        raise ValueError(
            "El dataset nacional contiene casos inválidos."
        )

    if datos["fecha_inicio_semana"].duplicated().any():
        raise ValueError(
            "El dataset nacional contiene semanas duplicadas."
        )

    return datos.sort_values(
        "fecha_inicio_semana"
    ).reset_index(drop=True)


def obtener_referencia_ann(fecha):
    """Obtiene clima y umbral histórico desde el dataset ANN.

    Para fechas existentes utiliza el registro original.
    Para fechas futuras utiliza referencias de las mismas
    semanas epidemiológicas de los últimos cinco años disponibles.
    """

    fecha = pd.Timestamp(fecha).normalize()
    referencias = cargar_datos_ann()

    coincidencia = referencias.loc[
        referencias["fecha_inicio_semana"] == fecha
    ]

    if not coincidencia.empty:
        fila = coincidencia.iloc[0]

        return {
            "lluvia_14d_mm": float(
                fila["lluvia_14d_mm"]
            ),
            "temp_max": float(
                fila["temp_max"]
            ),
            "umbral_historico": float(
                fila["casos_umbral_endemico"]
            ),
            "origen": "registro_historico_ann",
        }

    _, semana_epi = anio_semana_epi(fecha.date())

    anteriores = referencias.loc[
        (referencias["semana_epi"] == semana_epi)
        & (referencias["anio"] < fecha.year)
    ].copy()

    if anteriores.empty:
        raise ValueError(
            "No hay referencias históricas anteriores "
            f"para la semana epidemiológica {semana_epi}."
        )

    ultimos_anios = sorted(
        anteriores["anio"].unique(),
        reverse=True,
    )[:5]

    anteriores = anteriores.loc[
        anteriores["anio"].isin(ultimos_anios)
    ]

    return {
        "lluvia_14d_mm": float(
            anteriores["lluvia_14d_mm"].median()
        ),
        "temp_max": float(
            anteriores["temp_max"].median()
        ),
        "umbral_historico": None,
        "origen": "referencia_climatica_historica_ann",
    }


def calcular_umbral_endemico(fecha):
    """Calcula el percentil 75 de casos históricos de cuatro semanas.

    Utiliza una ventana de cinco años anteriores y requiere
    al menos tres años completos disponibles.
    """

    fecha = pd.Timestamp(fecha).normalize()

    nacional = cargar_casos_nacionales()

    _, semana_objetivo = anio_semana_epi(
        fecha.date()
    )

    serie = nacional.set_index(
        "fecha_inicio_semana"
    )["casos_dengue"]

    acumulados = []

    for anio in range(fecha.year - 5, fecha.year):

        fechas_anio = nacional.loc[
            nacional["fecha_inicio_semana"].dt.year == anio,
            "fecha_inicio_semana",
        ]

        inicio = None

        for fecha_candidata in fechas_anio:
            _, semana = anio_semana_epi(
                fecha_candidata.date()
            )

            if semana == semana_objetivo:
                inicio = fecha_candidata
                break

        if inicio is None:
            continue

        fechas_4_semanas = [
            inicio + pd.Timedelta(weeks=i)
            for i in range(4)
        ]

        if not all(
            fecha_semana in serie.index
            for fecha_semana in fechas_4_semanas
        ):
            continue

        total = float(
            serie.loc[fechas_4_semanas].sum()
        )

        acumulados.append(total)

    if len(acumulados) < 3:
        raise ValueError(
            "No existen al menos tres años históricos "
            "completos para calcular el umbral endémico "
            f"de la semana {semana_objetivo}."
        )

    return {
        "umbral": float(
            np.percentile(acumulados, 75)
        ),
        "anios_utilizados": len(acumulados),
        "origen": "percentil_75_casos_4_semanas_historicas",
    }


def preparar_contexto_ann(
    casos_pred_4sem,
    fecha_referencia,
):
    """Prepara las variables que necesita la ANN.

    Parámetros:
        casos_pred_4sem:
            Total de casos pronosticados por el LSTM
            para las siguientes cuatro semanas.

        fecha_referencia:
            Fecha de inicio de la primera semana
            pronosticada por el LSTM.

    No ejecuta el LSTM ni realiza inferencia de la ANN.
    """

    fecha = pd.Timestamp(
        fecha_referencia
    ).normalize()

    if pd.isna(fecha):
        raise ValueError(
            "La fecha de referencia es inválida."
        )

    casos = float(casos_pred_4sem)

    if not np.isfinite(casos) or casos < 0:
        raise ValueError(
            "Los casos pronosticados deben ser "
            "un número finito y no negativo."
        )

    if not casos.is_integer():
        raise ValueError(
            "El total de casos pronosticados debe ser entero."
        )

    casos = int(casos)

    # Semana epidemiológica calculada automáticamente.
    _, semana_epi = anio_semana_epi(
        fecha.date()
    )

    # Variables climáticas del propio dataset ANN.
    referencia_ann = obtener_referencia_ann(
        fecha
    )

    lluvia_14d = referencia_ann["lluvia_14d_mm"]
    temp_max = referencia_ann["temp_max"]

    # Umbral histórico o cálculo para fecha futura.
    if referencia_ann["umbral_historico"] is not None:
        umbral = referencia_ann["umbral_historico"]
        origen_umbral = "umbral_original_vectoria_acciones_csv"
    else:
        calculo = calcular_umbral_endemico(
            fecha
        )

        umbral = calculo["umbral"]
        origen_umbral = calculo["origen"]

    valores = [
        casos,
        umbral,
        lluvia_14d,
        temp_max,
        semana_epi,
    ]

    if not np.isfinite(valores).all():
        raise ValueError(
            "Las variables de ANN contienen valores inválidos."
        )

    if umbral < 0 or lluvia_14d < 0:
        raise ValueError(
            "El umbral y la lluvia no pueden ser negativos."
        )

    return {
        "casos_pred_4sem": casos,
        "umbral_endemico": round(umbral, 3),
        "lluvia_14d_mm": round(lluvia_14d, 3),
        "temp_max": round(temp_max, 3),
        "semana_epi": int(semana_epi),
        "fecha_referencia": fecha.date().isoformat(),
        "origen_clima": referencia_ann["origen"],
        "origen_umbral": origen_umbral,
        "advertencia": (
            "Los casos pronosticados proceden del LSTM. "
            "La lluvia y temperatura proceden de referencias "
            "históricas de vectoria_acciones.csv. "
            "El umbral utiliza el valor original para fechas "
            "disponibles o el percentil 75 de acumulados "
            "históricos de cuatro semanas para fechas futuras. "
            "Las estimaciones no son observaciones meteorológicas "
            "reales ni umbrales epidemiológicos oficiales."
        ),
    }
