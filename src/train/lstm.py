"""Módulo de entrenamiento, evaluación y pronóstico con modelos recurrentes (SimpleRNN y LSTM)
para la predicción de casos de dengue en Costa Rica (VectorIA).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential, load_model

logger = logging.getLogger(__name__)

# Rutas y constantes principales del módulo
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ_PROYECTO / "data" / "raw" / "VectorIA-datos"
RUTA_PROCESADOS = RAIZ_PROYECTO / "data" / "processed"
RUTA_MODELOS = RAIZ_PROYECTO / "models"
RUTA_FIGURAS = RAIZ_PROYECTO / "reports" / "figures"

# Asegurar la existencia de directorios de salida
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)
RUTA_MODELOS.mkdir(parents=True, exist_ok=True)

VENTANA = 26
HORIZONTE = 4
COLUMNAS_CLIMA = ["lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]


def cargar_datos(ruta: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Carga el dataset de casos de dengue y variables climáticas.

    Ordena cronológicamente por 'fecha_inicio_semana' y reinicia los índices.

    Args:
        ruta: Ruta opcional al archivo CSV. Si no se indica, busca en las ubicaciones estándar.

    Returns:
        pd.DataFrame con los datos ordenados.
    """
    if ruta is None:
        candidatos = [
            RUTA_DATOS / "vectoria_dengue_nacional_real.csv",
            RAIZ_PROYECTO / "data" / "raw" / "vectoria_dengue_nacional_real.csv",
            RAIZ_PROYECTO / "data" / "vectoria_dengue_nacional_real.csv",
            RAIZ_PROYECTO / "data" / "VectorIA-datos" / "vectoria_dengue_nacional_real.csv",
            Path("data/raw/VectorIA-datos/vectoria_dengue_nacional_real.csv"),
            Path("data/raw/vectoria_dengue_nacional_real.csv"),
            Path("data/vectoria_dengue_nacional_real.csv"),
            Path("vectoria_dengue_nacional_real.csv"),
        ]
        archivo_encontrado = None
        for cand in candidatos:
            if cand.exists():
                archivo_encontrado = cand
                break

        if archivo_encontrado is None:
            archivos_glob = list(RAIZ_PROYECTO.glob("**/vectoria_dengue_nacional_real.csv"))
            if archivos_glob:
                archivo_encontrado = archivos_glob[0]
            else:
                home = Path.home()
                archivos_home = list((home / "Downloads").glob("**/vectoria_dengue_nacional_real.csv"))
                if archivos_home:
                    archivo_encontrado = archivos_home[0]
                else:
                    raise FileNotFoundError(
                        f"No se encontró 'vectoria_dengue_nacional_real.csv'. "
                        f"Ubique el archivo en {RUTA_DATOS} o pase la ruta como argumento."
                    )
        ruta_csv = archivo_encontrado
    else:
        ruta_csv = Path(ruta)
        if not ruta_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv}")

    df = pd.read_csv(ruta_csv, parse_dates=["fecha_inicio_semana"])
    df = df.sort_values("fecha_inicio_semana").reset_index(drop=True)
    return df


def columnas_features() -> List[str]:
    """Retorna los nombres de las variables (features) utilizadas por los modelos."""
    return [
        "log_casos",
        "lluvia_mm",
        "temp_max_c",
        "temp_min_c",
        "humedad_rel_pct",
        "humedad_lag1",
        "lluvia_lag3",
        "temp_min_lag8",
        "temp_max_lag12",
        "sem_sin",
        "sem_cos",
    ]


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula log(1+casos), rezagos climáticos y codificación cíclica de la semana epidemiológica.

    Descarta las primeras 12 filas debido al rezago máximo (12 semanas).

    Args:
        df: DataFrame original ordenado cronológicamente.

    Returns:
        DataFrame con las nuevas columnas calculadas y sin valores nulos por rezago.
    """
    df_feat = df.copy()

    # Transformación logarítmica de casos
    df_feat["log_casos"] = np.log1p(df_feat["casos_dengue"])

    # Rezagos climáticos
    df_feat["humedad_lag1"] = df_feat["humedad_rel_pct"].shift(1)
    df_feat["lluvia_lag3"] = df_feat["lluvia_mm"].shift(3)
    df_feat["temp_min_lag8"] = df_feat["temp_min_c"].shift(8)
    df_feat["temp_max_lag12"] = df_feat["temp_max_c"].shift(12)

    # Codificación cíclica de semana epidemiológica
    semana = df_feat["semana_epi"].astype(float)
    df_feat["sem_sin"] = np.sin(2 * np.pi * semana / 52.0)
    df_feat["sem_cos"] = np.cos(2 * np.pi * semana / 52.0)

    # Eliminar filas con nulos introducidos por los shifts iniciales (12 semanas)
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat


def preparar_datos(
    df: pd.DataFrame,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    ajustar_con_todo: bool = False,
) -> Dict:
    """Escala las variables, construye las ventanas deslizantes 3D y define las particiones temporales.

    Particiones:
        - train: semanas objetivo en <= 2021
        - val: semanas objetivo en 2022
        - test: semanas objetivo en 2023-2024

    Args:
        df: DataFrame con los datos originales.
        ventana: Longitud de la secuencia de entrada (por defecto 26 semanas).
        horizonte: Longitud de la secuencia objetivo a predecir (por defecto 4 semanas).
        ajustar_con_todo: Si es True, entrena los escaladores con toda la serie para el modelo final.

    Returns:
        Diccionario con X, y, máscaras de partición, escaladores y metadatos.
    """
    df_feat = crear_features(df)
    features = columnas_features()

    n = len(df_feat)
    n_ventanas = n - ventana - horizonte + 1
    if n_ventanas <= 0:
        raise ValueError("El DataFrame no cuenta con suficientes observaciones para la ventana y horizonte dados.")

    idx_obj = np.arange(ventana, n - horizonte + 1)
    anios_feat = df_feat["anio"].to_numpy()

    mask_train = np.zeros(n_ventanas, dtype=bool)
    mask_val = np.zeros(n_ventanas, dtype=bool)
    mask_test = np.zeros(n_ventanas, dtype=bool)

    for i, idx in enumerate(idx_obj):
        anios_obj = anios_feat[idx : idx + horizonte]
        if np.all(anios_obj <= 2021):
            mask_train[i] = True
        elif np.all(anios_obj == 2022):
            mask_val[i] = True
        elif np.all((anios_obj >= 2023) & (anios_obj <= 2024)):
            mask_test[i] = True

    esc_x = MinMaxScaler(feature_range=(0, 1))
    esc_y = MinMaxScaler(feature_range=(0, 1))

    if ajustar_con_todo:
        esc_x.fit(df_feat[features])
        esc_y.fit(df_feat[["log_casos"]])
        mask_train = np.ones(n_ventanas, dtype=bool)
        mask_val = np.zeros(n_ventanas, dtype=bool)
        mask_test = np.zeros(n_ventanas, dtype=bool)
    else:
        filtro_train = df_feat["anio"] <= 2021
        esc_x.fit(df_feat.loc[filtro_train, features])
        esc_y.fit(df_feat.loc[filtro_train, ["log_casos"]])

    X_mat = esc_x.transform(df_feat[features])
    y_vec = esc_y.transform(df_feat[["log_casos"]]).flatten()

    X = np.empty((n_ventanas, ventana, len(features)), dtype=np.float32)
    y = np.empty((n_ventanas, horizonte), dtype=np.float32)

    for i in range(n_ventanas):
        X[i] = X_mat[i : i + ventana]
        y[i] = y_vec[i + ventana : i + ventana + horizonte]

    return {
        "X": X,
        "y": y,
        "df_feat": df_feat,
        "idx_obj": idx_obj,
        "masks": {
            "train": mask_train,
            "val": mask_val,
            "test": mask_test,
        },
        "esc_x": esc_x,
        "esc_y": esc_y,
        "ventana": ventana,
        "horizonte": horizonte,
        "features": features,
        "df_original": df,
    }


def casos_reales_ventanas(datos: Dict, mask: np.ndarray) -> np.ndarray:
    """Extrae la matriz de casos reales en escala original para las ventanas de la máscara dada."""
    idx_obj = datos["idx_obj"][mask]
    casos = datos["df_feat"]["casos_dengue"].to_numpy()
    horizonte = datos["horizonte"]

    reales = np.empty((len(idx_obj), horizonte), dtype=np.float64)
    for i, idx in enumerate(idx_obj):
        reales[i] = casos[idx : idx + horizonte]
    return reales


def baseline_estacional(datos: Dict, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula las predicciones del baseline estacional (misma semana epidemiológica del año anterior).

    Returns:
        base_est: Matriz de predicciones (N, horizonte).
        ok: Array booleano que indica las ventanas con histórico disponible completo.
    """
    df = datos.get("df_original", datos["df_feat"])
    idx_obj = datos["idx_obj"][mask]
    anios = datos["df_feat"]["anio"].to_numpy()
    semanas = datos["df_feat"]["semana_epi"].to_numpy()
    horizonte = datos["horizonte"]

    mapa_casos = {(row.anio, row.semana_epi): row.casos_dengue for row in df.itertuples()}

    base_est = np.zeros((len(idx_obj), horizonte), dtype=np.float64)
    ok = np.ones(len(idx_obj), dtype=bool)

    for i, idx in enumerate(idx_obj):
        for h in range(horizonte):
            a_obj = anios[idx + h]
            s_obj = semanas[idx + h]
            clave_prev = (a_obj - 1, s_obj)
            if clave_prev in mapa_casos:
                base_est[i, h] = mapa_casos[clave_prev]
            else:
                ok[i] = False

    return base_est, ok


def baseline_persistencia(datos: Dict, mask: np.ndarray) -> np.ndarray:
    """Calcula el baseline de persistencia repitiendo los casos de la última semana observada."""
    idx_obj = datos["idx_obj"][mask]
    casos = datos["df_feat"]["casos_dengue"].to_numpy()
    horizonte = datos["horizonte"]

    base_per = np.empty((len(idx_obj), horizonte), dtype=np.float64)
    for i, idx in enumerate(idx_obj):
        ult_caso = casos[idx - 1]
        base_per[i] = np.full(horizonte, ult_caso)
    return base_per


def rmse(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """Calcula la raíz del error cuadrático medio."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """Calcula el error absoluto medio."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(yt - yp)))


def construir_simplernn(
    n_features: int,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    unidades: int = 32,
    lr: float = 0.001,
) -> tf.keras.Model:
    """Construye y compila el modelo base SimpleRNN(32) -> Dense(4)."""
    modelo = Sequential([
        Input(shape=(ventana, n_features)),
        SimpleRNN(unidades),
        Dense(horizonte),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


def construir_lstm(
    n_features: int,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    u1: int = 64,
    u2: int = 32,
    dropout: float = 0.2,
    lr: float = 0.001,
) -> tf.keras.Model:
    """Construye y compila el modelo principal LSTM(64) -> LSTM(32) -> Dropout(0.2) -> Dense(4)."""
    modelo = Sequential([
        Input(shape=(ventana, n_features)),
        LSTM(u1, return_sequences=True),
        LSTM(u2),
        Dropout(dropout),
        Dense(horizonte),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


def entrenar(
    modelo: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 16,
    patience: int = 20,
    verbose: int = 0,
) -> tf.keras.callbacks.History:
    """Entrena el modelo neuronal aplicando EarlyStopping sobre la pérdida de validación si se proporciona."""
    callbacks = []
    val_data = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_data = (X_val, y_val)
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            )
        )

    return modelo.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )


def evaluar_en(datos: Dict, modelo: tf.keras.Model, particion: str) -> Dict:
    """Evalúa un modelo sobre una partición ('val' o 'test') calculando predicciones en escala real y métricas."""
    mask = datos["masks"][particion]
    X_eval = datos["X"][mask]
    y_real = casos_reales_ventanas(datos, mask)

    y_pred_scaled = modelo.predict(X_eval, verbose=0)
    y_pred_log = datos["esc_y"].inverse_transform(y_pred_scaled)
    y_pred_casos = np.expm1(y_pred_log)
    y_pred_casos = np.clip(y_pred_casos, 0, None)

    metricas = {
        "RMSE": rmse(y_real, y_pred_casos),
        "MAE": mae(y_real, y_pred_casos),
    }
    for h in range(1, datos["horizonte"] + 1):
        metricas[f"RMSE_h{h}"] = rmse(y_real[:, h - 1], y_pred_casos[:, h - 1])
        metricas[f"MAE_h{h}"] = mae(y_real[:, h - 1], y_pred_casos[:, h - 1])

    return {
        "pred": y_pred_casos,
        "real": y_real,
        "metricas": metricas,
    }


def guardar_artefactos(
    modelo: tf.keras.Model,
    esc_x: MinMaxScaler,
    esc_y: MinMaxScaler,
    nombre: str,
    metadata: Optional[Dict] = None,
    ruta: Optional[Union[str, Path]] = None,
) -> None:
    """Guarda el modelo (.keras), los escaladores (.joblib) y los metadatos (.json)."""
    if ruta is None:
        ruta = RUTA_MODELOS
    ruta_dir = Path(ruta)
    ruta_dir.mkdir(parents=True, exist_ok=True)

    modelo.save(ruta_dir / f"{nombre}.keras")
    joblib.dump(esc_x, ruta_dir / f"{nombre}_esc_x.joblib")
    joblib.dump(esc_y, ruta_dir / f"{nombre}_esc_y.joblib")

    if metadata is not None:
        def _convertir(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convertir(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convertir(v) for v in obj]
            return obj

        with open(ruta_dir / f"{nombre}_info.json", "w", encoding="utf-8") as f:
            json.dump(_convertir(metadata), f, indent=4, ensure_ascii=False)


def pronosticar_futuro(
    modelo: tf.keras.Model,
    df: pd.DataFrame,
    esc_x: MinMaxScaler,
    esc_y: MinMaxScaler,
    hasta_anio: int = 2026,
    clima_futuro: Optional[pd.DataFrame] = None,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
) -> pd.DataFrame:
    """Genera pronósticos recursivos semana a semana hasta el final del año indicado."""
    df_base = df.sort_values("fecha_inicio_semana").reset_index(drop=True).copy()
    ult_fecha = df_base["fecha_inicio_semana"].max()

    # Calcular climatología histórica por semana epidemiológica
    climatologia = (
        df_base.groupby("semana_epi")[COLUMNAS_CLIMA]
        .mean()
        .to_dict(orient="index")
    )

    # Generar calendario de semanas futuras
    fechas_futuras = []
    fecha_cursor = ult_fecha + pd.Timedelta(days=7)
    while True:
        anio_cursor = fecha_cursor.year
        if anio_cursor > hasta_anio:
            break
        fechas_futuras.append(fecha_cursor)
        fecha_cursor += pd.Timedelta(days=7)

    df_futuras = pd.DataFrame({"fecha_inicio_semana": fechas_futuras})
    # Obtener semana epidemiológica según ISO calendar
    df_futuras["anio"] = df_futuras["fecha_inicio_semana"].dt.isocalendar().year
    df_futuras["semana_epi"] = df_futuras["fecha_inicio_semana"].dt.isocalendar().week

    for col in COLUMNAS_CLIMA:
        if clima_futuro is not None and col in clima_futuro.columns:
            df_futuras[col] = clima_futuro[col].to_numpy()
        else:
            df_futuras[col] = df_futuras["semana_epi"].map(
                lambda s: climatologia.get(int(s), climatologia.get(1, {}))[col]
            )

    df_futuras["casos_dengue"] = np.nan
    df_futuras["clima_usado"] = "climatologia_historica" if clima_futuro is None else "clima_provisto"

    df_extendido = pd.concat([df_base, df_futuras], ignore_index=True)
    idx_inicio_futuro = len(df_base)

    features = columnas_features()

    for idx_fut in range(idx_inicio_futuro, len(df_extendido)):
        sub_df = df_extendido.iloc[:idx_fut + 1].copy()
        sub_feat = crear_features(sub_df)
        if len(sub_feat) < ventana:
            continue

        ventana_df = sub_feat.iloc[-ventana:][features]
        ventana_scaled = esc_x.transform(ventana_df).reshape(1, ventana, len(features))

        pred_scaled = modelo.predict(ventana_scaled, verbose=0)
        pred_log = esc_y.inverse_transform(pred_scaled)
        pred_casos = np.expm1(pred_log)
        pred_casos = np.clip(pred_casos, 0, None)

        caso_siguiente = float(pred_casos[0, 0])
        df_extendido.at[idx_fut, "casos_dengue"] = caso_siguiente

    df_resultado = df_extendido.iloc[idx_inicio_futuro:].copy().reset_index(drop=True)
    df_resultado["casos_pred"] = df_resultado["casos_dengue"]
    return df_resultado


def tabla_evaluacion(datos: Dict, preds_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Construye el DataFrame de evaluación sobre el conjunto de prueba para el horizonte de 1 semana."""
    idx_test = datos["idx_obj"][datos["masks"]["test"]]
    df_feat = datos["df_feat"]

    filas = {
        "fecha_inicio_semana": df_feat["fecha_inicio_semana"].iloc[idx_test].to_numpy(),
        "anio": df_feat["anio"].iloc[idx_test].to_numpy(),
        "semana_epi": df_feat["semana_epi"].iloc[idx_test].to_numpy(),
        "casos_reales": df_feat["casos_dengue"].iloc[idx_test].to_numpy(),
        "pred_SimpleRNN": preds_dict["SimpleRNN"][:, 0],
        "pred_LSTM": preds_dict["LSTM"][:, 0],
        "periodo": "evaluacion_" + df_feat["anio"].iloc[idx_test].astype(str).to_numpy(),
        "clima_usado": "observado_real",
    }
    df_eval = pd.DataFrame(filas)
    df_eval["casos_acum_4sem_LSTM"] = np.nan
    return df_eval
