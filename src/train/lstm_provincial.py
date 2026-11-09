"""Entrenamiento del bonus LSTM provincial (datos sintéticos 2014-2025).

Ejecutar desde la raíz de VectorIA:
    python -m src.train.lstm_provincial
No modifica el modelo nacional ni el CSV original.
"""
from __future__ import annotations

import json
import os
import random
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'vectoria_dengue_semanal.csv'
OUT_DIR = BASE_DIR / 'models' / 'LSTM_Provincial'
COLUMNS = ['fecha_inicio_semana', 'anio', 'semana_epi', 'provincia', 'region', 'poblacion',
           'casos_dengue', 'incidencia_100k', 'lluvia_mm', 'temp_max', 'temp_min', 'humedad']
PROVINCIAS = ['Alajuela', 'Cartago', 'Guanacaste', 'Heredia', 'Limón', 'Puntarenas', 'San José']
CLIMA = ['lluvia_mm', 'temp_max', 'temp_min', 'humedad']
BASE_FEATURES = ['log_casos', *CLIMA, 'semana_sin', 'semana_cos']
FEATURES = BASE_FEATURES + [f'prov_{i}' for i in range(7)]
VENTANA = 26
HORIZONTE = 4
SEMILLA = 42


def leer_y_validar(ruta: Path = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Detecta CSV sin encabezado (como el facilitado) o con encabezado.

    Imputa SOLO clima; nunca inventa casos. Devuelve informe reproducible.
    """
    if not Path(ruta).is_file():
        raise FileNotFoundError(f'No existe {ruta}. Coloque el CSV original en data/raw/.')
    primer = Path(ruta).read_text(
        encoding="utf-8-sig"
    ).splitlines()[0]

    tiene_encabezado = primer.lower().startswith(
        "fecha_inicio_semana,"
    )

    df = pd.read_csv(
        ruta,
        header=0 if tiene_encabezado else None,
        encoding="utf-8-sig"
    )

    # Verificar que tenga las 12 columnas esperadas.
    if len(df.columns) != len(COLUMNS):
        raise ValueError(
            f"Se esperaban {len(COLUMNS)} columnas, "
            f"pero se encontraron {len(df.columns)}."
        )

    # Asignar nombres estandarizados según el orden del dataset.
    df.columns = COLUMNS
    df = df[COLUMNS].copy()
    df['fecha_inicio_semana'] = pd.to_datetime(df['fecha_inicio_semana'], errors='raise')
    numericas = ['anio', 'semana_epi', 'poblacion', 'casos_dengue', 'incidencia_100k', *CLIMA]
    for c in numericas:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df['provincia'].isna().any() or set(df['provincia']) != set(PROVINCIAS):
        raise ValueError(f'Las provincias deben ser exactamente las 7 conocidas: {df.provincia.unique().tolist()}')
    if df.duplicated(['provincia', 'fecha_inicio_semana']).any():
        raise ValueError('Hay provincia-fecha duplicadas; no se eliminarán silenciosamente')
    for c in ['anio', 'semana_epi', 'poblacion', 'casos_dengue']:
        if df[c].isna().any():
            raise ValueError(f'Valores ausentes en campo esencial: {c}')
    if (df['casos_dengue'] < 0).any() or (df['casos_dengue'] % 1 != 0).any():
        raise ValueError('Casos de dengue negativos o no enteros')
    if df['fecha_inicio_semana'].dt.dayofweek.ne(6).any():
        raise ValueError('Las fechas deben corresponder a domingo (intervalos semanales)')
    if df['fecha_inicio_semana'].dt.year.ne(df['anio']).any():
        raise ValueError('Columna anio no coincide con la fecha')
    if (~df['semana_epi'].between(1, 53)).any():
        raise ValueError('Semanas epidemiológicas fuera de 1..53')
    if (~np.isfinite(df[['poblacion', 'casos_dengue']].to_numpy(dtype=float))).any():
        raise ValueError('Población o casos no finitos')
    df = df.sort_values(['provincia', 'fecha_inicio_semana']).reset_index(drop=True)
    for prov, grp in df.groupby('provincia'):
        esperadas = pd.date_range(grp['fecha_inicio_semana'].iloc[0], grp['fecha_inicio_semana'].iloc[-1], freq='W-SUN')
        if len(esperadas) != len(grp) or not grp['fecha_inicio_semana'].reset_index(drop=True).equals(pd.Series(esperadas, name='fecha_inicio_semana')):
            raise ValueError(f'Faltan o sobran semanas en {prov}')
    nulos_antes = df[CLIMA].isna().sum().to_dict()
    # Interpolación temporal SOLO para clima; los casos permanecen idénticos.
    for c in CLIMA:
        df[c] = df.groupby('provincia')[c].transform(lambda s: s.interpolate(limit_direction='both'))
        if df[c].isna().any():
            raise ValueError(f'No se pudo imputar {c}')
    if not np.isfinite(df[CLIMA].to_numpy(dtype=float)).all():
        raise ValueError('Datos climáticos inválidos después de limpieza')
    if (df['lluvia_mm'] < 0).any() or (~df['humedad'].between(0, 100)).any() or (df['temp_max'] < df['temp_min']).any():
        raise ValueError('Valores climáticos físicamente inconsistentes')
    # Las semanas epidemiológicas del CSV no siguen necesariamente ISO 8601.
    # Para periodicidad del modelo se deriva una fase de fecha consistente (semana ISO).
    semanas = df['fecha_inicio_semana'].dt.isocalendar().week.astype(int)
    df['semana_sin'] = np.sin(2 * np.pi * (semanas - 1) / 52.1775)
    df['semana_cos'] = np.cos(2 * np.pi * (semanas - 1) / 52.1775)
    df['log_casos'] = np.log1p(df['casos_dengue'].astype(float))
    for i, prov in enumerate(PROVINCIAS):
        df[f'prov_{i}'] = (df['provincia'] == prov).astype(float)
    reporte = {'filas': len(df), 'provincias': df['provincia'].value_counts().to_dict(),
               'inicio': df['fecha_inicio_semana'].min().date().isoformat(),
               'fin': df['fecha_inicio_semana'].max().date().isoformat(),
               'nulos_clima_originales': {k: int(v) for k, v in nulos_antes.items()},
               'duplicados_provincia_fecha': 0, 'observacion': 'Datos provinciales sintéticos; no son registros oficiales.'}
    return df, reporte


def crear_ventanas(df: pd.DataFrame, escalador_x: StandardScaler, escalador_y: StandardScaler):
    X, y, fechas, provincias, bases = [], [], [], [], []
    for prov, grupo in df.groupby('provincia', sort=True):
        grupo = grupo.sort_values('fecha_inicio_semana').reset_index(drop=True)
        x = escalador_x.transform(grupo[FEATURES].to_numpy(dtype=float))
        z = escalador_y.transform(grupo[['log_casos']].to_numpy(dtype=float)).reshape(-1)
        casos = grupo['casos_dengue'].to_numpy(dtype=float)
        fechas_prov = grupo['fecha_inicio_semana'].tolist()
        for fin in range(VENTANA, len(grupo) - HORIZONTE + 1):
            X.append(x[fin-VENTANA:fin]); y.append(z[fin:fin+HORIZONTE])
            fechas.append(fechas_prov[fin + HORIZONTE - 1])
            provincias.append(prov)
            # Baseline estacional: mismas 4 semanas de 52 semanas anteriores.
            bases.append(casos[fin-52:fin-52+HORIZONTE] if fin >= 52 else np.full(HORIZONTE, np.nan))
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            pd.to_datetime(fechas), np.asarray(provincias), np.asarray(bases, dtype=float))


def evaluar(modelo, X, y, base, sy):
    if not len(X):
        raise ValueError('El split no tiene ventanas')
    pred = modelo.predict(X, batch_size=256, verbose=0)
    pred_casos = np.maximum(0, np.expm1(sy.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)))
    real_casos = np.maximum(0, np.expm1(sy.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)))
    met = {'MAE': round(float(mean_absolute_error(real_casos.ravel(), pred_casos.ravel())), 3),
           'RMSE': round(float(np.sqrt(mean_squared_error(real_casos.ravel(), pred_casos.ravel()))), 3),
           'ventanas': int(len(X))}
    validas = np.isfinite(base).all(axis=1)
    if validas.any():
        met['baseline_estacional_MAE'] = round(float(mean_absolute_error(real_casos[validas].ravel(), base[validas].ravel())), 3)
        met['baseline_estacional_RMSE'] = round(float(np.sqrt(mean_squared_error(real_casos[validas].ravel(), base[validas].ravel()))), 3)
    return met


def entrenar(epochs=40, batch_size=64):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEMILLA)
    np.random.seed(SEMILLA); random.seed(SEMILLA)
    df, reporte = leer_y_validar()
    # Train <=2023; validación año 2024; test año 2025. Sin fuga temporal en targets.
    train = df[df['fecha_inicio_semana'] <= '2023-12-31']
    sx = StandardScaler().fit(train[FEATURES].to_numpy(dtype=float))
    sy = StandardScaler().fit(train[['log_casos']].to_numpy(dtype=float))
    X, y, fechas, provincias, baseline = crear_ventanas(df, sx, sy)
    a = fechas <= pd.Timestamp('2023-12-31')
    b = (fechas >= pd.Timestamp('2024-01-01')) & (fechas <= pd.Timestamp('2024-12-31'))
    c = (fechas >= pd.Timestamp('2025-01-01')) & (fechas <= pd.Timestamp('2025-12-31'))
    if min(a.sum(), b.sum(), c.sum()) == 0:
        raise ValueError('Faltan ventanas para entrenar, validar o probar')
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input((VENTANA, len(FEATURES))),
        tf.keras.layers.LSTM(48, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(24),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(HORIZONTE),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=4, min_lr=1e-5)]
    historial = modelo.fit(X[a], y[a], validation_data=(X[b], y[b]), epochs=int(epochs),
                          batch_size=int(batch_size), callbacks=callback, verbose=2, shuffle=False)
    metricas = {'entrenamiento': evaluar(modelo, X[a], y[a], baseline[a], sy),
                'validacion_2024': evaluar(modelo, X[b], y[b], baseline[b], sy),
                'prueba_2025': evaluar(modelo, X[c], y[c], baseline[c], sy)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modelo.save(OUT_DIR / 'lstm_provincial.keras')
    joblib.dump({'x': sx, 'y': sy}, OUT_DIR / 'scaler_provincial.joblib')
    config = {'features': FEATURES, 'provincias': PROVINCIAS, 'ventana': VENTANA,
              'horizonte': HORIZONTE, 'ultima_fecha': df.fecha_inicio_semana.max().date().isoformat(),
              'dataset': DATA_PATH.name, 'tipo_datos': 'sinteticos', 'seed': SEMILLA,
              'periodos': {'train': 'hasta 2023', 'val': '2024', 'test': '2025'}}
    (OUT_DIR / 'config_provincial.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT_DIR / 'metricas_provincial.json').write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT_DIR / 'reporte_limpieza.json').write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nMÉTRICAS\n' + json.dumps(metricas, ensure_ascii=False, indent=2))
    print('\nArchivos creados en', OUT_DIR)
    return metricas, historial.history


if __name__ == '__main__':
    entrenar()
