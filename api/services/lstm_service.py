
from datetime import date
from functools import lru_cache

import pandas as pd

from src.train.lstm import (
    VENTANA,
    cargar_datos,
    cargar_artefactos,
    predecir_4_semanas,
    pronosticar_futuro,
)


# Modelo nacional entrenado por Caleb.
NOMBRE_MODELO = "lstm_dengue_final"
HORIZONTE = 4


@lru_cache(maxsize=1)
def obtener_recursos():
    """
    Carga una sola vez el modelo, sus escaladores
    y el dataset historico nacional.
    """
    modelo, esc_x, esc_y = cargar_artefactos(
        NOMBRE_MODELO
    )

    historial = cargar_datos()

    if historial.empty:
        raise ValueError("El dataset historico esta vacio.")

    historial = historial.sort_values(
        "fecha_inicio_semana"
    ).reset_index(drop=True)

    return modelo, esc_x, esc_y, historial


@lru_cache(maxsize=1)
def obtener_pronostico_futuro():
    """
    Genera una sola vez las predicciones recursivas
    para 2025 y 2026, sin volver a entrenar.
    """
    modelo, esc_x, esc_y, historial = obtener_recursos()

    pronostico = pronosticar_futuro(
        modelo=modelo,
        df=historial,
        esc_x=esc_x,
        esc_y=esc_y,
        hasta_anio=2026,
    )

    return pronostico.sort_values(
        "fecha_inicio_semana"
    ).reset_index(drop=True)


def predecir_dengue(fecha: date):
    """
    Predice los casos nacionales de dengue para
    las cuatro semanas posteriores a la fecha.

    Fechas historicas:
        Utiliza solamente datos observados
        hasta la fecha seleccionada.

    Fechas futuras:
        Utiliza el pronostico recursivo del LSTM
        generado hasta finales de 2026.
    """

    modelo, esc_x, esc_y, historial = obtener_recursos()

    fecha_consulta = pd.Timestamp(fecha)
    ultima_fecha_real = historial[
        "fecha_inicio_semana"
    ].max()

    resultados = []
    metodo = None

    # -----------------------------------------
    # 1. Prediccion desde una fecha historica
    # -----------------------------------------
    if fecha_consulta <= ultima_fecha_real:

        historia_disponible = historial.loc[
            historial["fecha_inicio_semana"]
            <= fecha_consulta
        ].copy()

        # 26 semanas + 12 para calcular rezagos
        minimo = VENTANA + 12

        if len(historia_disponible) < minimo:
            raise ValueError(
                f"Se requieren al menos {minimo} "
                "semanas historicas para predecir."
            )

        ultima_semana = historia_disponible[
            "fecha_inicio_semana"
        ].max()

        predicciones = predecir_4_semanas(
            modelo=modelo,
            hist_df=historia_disponible,
            esc_x=esc_x,
            esc_y=esc_y,
        )

        for i, casos in enumerate(predicciones):
            fecha_pred = ultima_semana + pd.Timedelta(
                weeks=i + 1
            )

            resultados.append({
                "semana": i + 1,
                "fecha": fecha_pred.date().isoformat(),
                "casos_estimados": max(
                    0, int(round(float(casos)))
                ),
            })

        metodo = "prediccion_desde_historial"

    # -----------------------------------------
    # 2. Prediccion desde fechas futuras
    # -----------------------------------------
    else:
        pronostico = obtener_pronostico_futuro()

        semanas_futuras = pronostico.loc[
            pronostico["fecha_inicio_semana"]
            > fecha_consulta
        ].head(HORIZONTE)

        if len(semanas_futuras) < HORIZONTE:
            raise ValueError(
                "No existen cuatro semanas futuras "
                "disponibles para esta fecha. "
                "El pronostico actual termina en 2026."
            )

        for i, (_, fila) in enumerate(
            semanas_futuras.iterrows(), start=1
        ):
            resultados.append({
                "semana": i,
                "fecha": fila[
                    "fecha_inicio_semana"
                ].date().isoformat(),
                "casos_estimados": max(
                    0,
                    int(round(float(fila["casos_pred"])))
                ),
            })

        metodo = "pronostico_recursivo"

    # -----------------------------------------
    # 3. Respuesta para FastAPI y Streamlit
    # -----------------------------------------
    return {
        "estado": "correcto",
        "modelo": "LSTM",
        "tipo": "nacional",
        "modelo_utilizado": NOMBRE_MODELO,
        "fecha_seleccionada": fecha.isoformat(),
        "ultima_fecha_historica": (
            ultima_fecha_real.date().isoformat()
        ),
        "metodo": metodo,
        "semanas_prediccion": HORIZONTE,
        "predicciones": resultados,
        "total_casos_estimados": sum(
            item["casos_estimados"]
            for item in resultados
        ),
        "clima_futuro": (
            "climatologia_historica"
            if metodo == "pronostico_recursivo"
            else "no_aplica"
        ),
    }
