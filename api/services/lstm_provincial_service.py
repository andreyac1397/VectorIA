"""Inferencia provincial 4 semanas. Independiente del LSTM nacional.

Para fechas de 2026 se avanza desde 2025-12-28 usando exclusivamente
predicciones previas y referencias climáticas semanales históricas.
El resultado NO es una observación ni un pronóstico climático de 2026.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train.lstm_provincial import (
    BASE_DIR, CLIMA, DATA_PATH, FEATURES, HORIZONTE, OUT_DIR, PROVINCIAS,
    VENTANA, leer_y_validar,
)


def _domingo_siguiente_o_actual(fecha: date) -> date:
    return fecha + timedelta(days=(6 - fecha.weekday()) % 7)


@lru_cache(maxsize=1)
def _recursos():
    from tensorflow.keras.models import load_model
    modelo_path = OUT_DIR / 'lstm_provincial.keras'
    scaler_path = OUT_DIR / 'scaler_provincial.joblib'
    config_path = OUT_DIR / 'config_provincial.json'
    for p in (modelo_path, scaler_path, config_path, DATA_PATH):
        if not p.is_file():
            raise FileNotFoundError(f'Archivo requerido no encontrado: {p}. Ejecute el entrenamiento primero.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('features') != FEATURES or config.get('provincias') != PROVINCIAS
            or config.get('ventana') != VENTANA or config.get('horizonte') != HORIZONTE):
        raise ValueError('Configuración provincial incompatible con este código')
    sc = joblib.load(scaler_path)  # Artefacto propio y confiable generado por el entrenamiento.
    sx, sy = sc['x'], sc['y']
    modelo = load_model(modelo_path, compile=False)
    if modelo.input_shape[-2:] != (VENTANA, len(FEATURES)) or modelo.output_shape[-1] != HORIZONTE:
        raise ValueError('Dimensiones del modelo provincial incompatibles')
    df, _ = leer_y_validar()
    if config['ultima_fecha'] != df.fecha_inicio_semana.max().date().isoformat():
        raise ValueError('El CSV original cambió después del entrenamiento. Vuelva a entrenar.')
    return modelo, sx, sy, df, config


def _fila_futura(provincia: str, fecha: date, casos: float, climatologia):
    semana = date.fromisoformat(fecha.isoformat()).isocalendar().week
    valores = {'log_casos': np.log1p(max(0.0, float(casos))),
               'semana_sin': np.sin(2 * np.pi * (semana - 1) / 52.1775),
               'semana_cos': np.cos(2 * np.pi * (semana - 1) / 52.1775)}
    for c in CLIMA:
        valores[c] = float(climatologia[(provincia, c)].get(semana, climatologia[(provincia, c)]['general']))
    for i, p in enumerate(PROVINCIAS):
        valores[f'prov_{i}'] = float(p == provincia)
    return valores


@lru_cache(maxsize=1)
def _pronostico_completo():
    """Calcula predicciones recursivas hasta 2027-01-24 una sola vez.

    Las predicciones que alimentan ventanas posteriores NO son datos reales.
    """
    modelo, sx, sy, df, config = _recursos()
    ultima = date.fromisoformat(config['ultima_fecha'])
    primera = ultima + timedelta(days=7)
    final = date(2027, 1, 24)
    if primera != date(2026, 1, 4):
        raise ValueError(f'Se esperaba el final histórico 2025-12-28; se encontró {ultima}')
    clima = {}
    for prov in PROVINCIAS:
        g = df[df.provincia == prov].copy()
        g['semana_iso'] = g.fecha_inicio_semana.dt.isocalendar().week.astype(int)
        for c in CLIMA:
            resumen = g.groupby('semana_iso')[c].median()
            clima[(prov, c)] = {int(k): float(v) for k, v in resumen.items()}
            clima[(prov, c)]['general'] = float(g[c].median())
    # Estado por provincia: cada ventana de 26 semanas con datos históricos o simulados.
    historiales = {p: df.loc[df.provincia == p, FEATURES].tail(VENTANA).to_numpy(dtype=float)
                  for p in PROVINCIAS}
    respuesta = {p: {} for p in PROVINCIAS}
    fecha_bloque = primera
    while fecha_bloque <= final:
        tensor = np.stack([sx.transform(historiales[p]) for p in PROVINCIAS]).astype(np.float32)
        pred_escaladas = np.asarray(modelo.predict(tensor, verbose=0), dtype=float)
        if pred_escaladas.shape != (len(PROVINCIAS), HORIZONTE) or not np.isfinite(pred_escaladas).all():
            raise RuntimeError('Predicciones provinciales no válidas')
        log_pred = sy.inverse_transform(pred_escaladas.reshape(-1, 1)).reshape(pred_escaladas.shape)
        casos_pred = np.clip(np.expm1(np.clip(log_pred, 0, np.log1p(1e6))), 0, 1e6)
        for j in range(HORIZONTE):
            fecha = fecha_bloque + timedelta(weeks=j)
            for i, prov in enumerate(PROVINCIAS):
                casos = int(np.rint(casos_pred[i, j]))
                respuesta[prov][fecha.isoformat()] = casos
                fila = _fila_futura(prov, fecha, casos, clima)
                historiales[prov] = np.vstack((historiales[prov][1:],
                                              np.array([fila[col] for col in FEATURES], dtype=float)))
        fecha_bloque += timedelta(weeks=HORIZONTE)
    return respuesta


def predecir_dengue_provincial(fecha: date | str, provincia: str | None = None) -> dict:
    """Primera semana solicitada = domingo coincidente o posterior a `fecha`.

    Si provincia es None, devuelve las siete para colorear el mapa.
    """
    if isinstance(fecha, str):
        fecha = date.fromisoformat(fecha)
    if not isinstance(fecha, date):
        raise ValueError('fecha debe tener formato YYYY-MM-DD')
    if not (date(2026, 1, 1) <= fecha <= date(2026, 12, 27)):
        raise ValueError('Seleccione una fecha entre 2026-01-01 y 2026-12-27')
    if provincia is not None and provincia not in PROVINCIAS:
        raise ValueError(f'Provincia inválida: {provincia}. Valores: {PROVINCIAS}')
    primera = _domingo_siguiente_o_actual(fecha)
    fechas = [(primera + timedelta(weeks=j)).isoformat() for j in range(HORIZONTE)]
    predecidas = _pronostico_completo()
    seleccion = PROVINCIAS if provincia is None else [provincia]
    provincias_resultado = []
    for prov in seleccion:
        semanas = [{'semana': i+1, 'fecha': f, 'casos_estimados': predecidas[prov][f]}
                   for i, f in enumerate(fechas)]
        provincias_resultado.append({'provincia': prov, 'total_4_semanas': sum(x['casos_estimados'] for x in semanas),
                                     'predicciones': semanas})
    return {'estado': 'correcto', 'modelo': 'LSTM provincial', 'tipo_datos': 'sinteticos',
            'fecha_solicitada': fecha.isoformat(), 'primera_semana': primera.isoformat(),
            'horizonte_semanas': HORIZONTE, 'provincia_solicitada': provincia,
            'resultados': provincias_resultado,
            'advertencia': ('Datos provinciales sintéticos. Predicciones recursivas para 2026 '
                            'basadas en 2014-2025; no son casos observados ni pronósticos oficiales. '
                            'La incertidumbre aumenta cuanto más se aleja la fecha de 2025.')}
