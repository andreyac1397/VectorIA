"""Módulo de entrenamiento, evaluación y pronóstico con modelos recurrentes (SimpleRNN, LSTM, GRU)
para la predicción de casos de dengue en Costa Rica (Proyecto VectorIA).

Este módulo proporciona el flujo de trabajo completo:
1. Carga y ordenamiento cronológico del dataset nacional con calendario epidemiológico ISO.
2. Ingeniería de variables: transformación logarítmica, rezagos climáticos biológicos
   y codificación armónica (seno/coseno) de la semana epidemiológica.
3. Construcción de tensores 3D con ventanas deslizantes temporales (Lookback = 26 semanas, Horizonte = 4 semanas).
4. Partición temporal estricta sin fuga de información (Train <= 2021, Val = 2022, Test = 2023-2024).
5. Cálculo de baselines de referencia (Estacional histórico y Persistencia).
6. Validación cruzada de ventana expansiva (Expanding Window CV) para selección rigurosa de candidatos (A–G).
7. Construcción y entrenamiento determinista de redes recurrentes (SimpleRNN, LSTM, GRU con residuo autorregresivo).
8. Evaluación en escala real de casos (RMSE, MAE y Skill Score multihorizonte).
9. Persistencia de artefactos unificados (.keras y _scalers.joblib) y API de inferencia para Streamlit.
10. Pronóstico recursivo hacia el futuro (2025–2026) con climatología estacional.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Add, Dense, Dropout, GRU, Input, Layer, LSTM, SimpleRNN
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.regularizers import l2

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y CONSTANTES GLOBALES
# ==============================================================================
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ_PROYECTO / "data" / "raw" / "VectorIA-datos"
RUTA_PROCESADOS = RAIZ_PROYECTO / "data" / "processed"
RUTA_MODELOS = RAIZ_PROYECTO / "models" / "LSTM"
RUTA_FIGURAS = RAIZ_PROYECTO / "reports" / "figures"

# Asegurar la existencia de las carpetas de salida en la estructura del proyecto
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)
RUTA_MODELOS.mkdir(parents=True, exist_ok=True)

# Parámetros temporales del problema epidemiológico
VENTANA = 26      # Semanas de historia pasada observada (Lookback: medio año)
HORIZONTE = 4     # Semanas futuras consecutivas a predecir (1 mes de alerta temprana)
COLUMNAS_CLIMA = ["lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]


# ==============================================================================
# 0. REPRODUCIBILIDAD Y CONTROL DETERMINISTA
# ==============================================================================
def fijar_semilla(seed: int = 42) -> None:
    """Fija la semilla en todos los generadores pseudoaleatorios y habilita determinismo en TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ==============================================================================
# 1. CALENDARIO EPIDEMIOLÓGICO Y CARGA DE DATOS
# ==============================================================================
def anio_semana_epi(fecha: Union[str, pd.Timestamp]) -> Tuple[int, int]:
    """Calcula el año y la semana epidemiológica bajo el estándar ISO con inicio en domingo.

    Dado que las semanas epidemiológicas en el dataset inician en domingo, se suma 1 día
    para alinearlas con el calendario ISO (lunes-domingo) estándar de Costa Rica.

    Args:
        fecha: Fecha de inicio de semana epidemiológica.

    Returns:
        Tupla (anio_epidemiologico, semana_epidemiologica).
    """
    ts = pd.to_datetime(fecha)
    iso = (ts + pd.Timedelta(days=1)).isocalendar()
    return int(iso.year), int(iso.week)


def cargar_datos(ruta: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Carga el dataset de casos de dengue y variables climáticas nacionales.

    Realiza una búsqueda del archivo 'vectoria_dengue_nacional_real.csv'
    en diversas rutas estándar del proyecto o en la carpeta Downloads del usuario.
    Ordena cronológicamente los registros por 'fecha_inicio_semana'.

    Args:
        ruta: Ruta opcional personalizada hacia el archivo CSV.

    Returns:
        pd.DataFrame con los datos limpios y ordenados en orden cronológico ascendente.
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


# ==============================================================================
# 2. INGENIERÍA DE VARIABLES (FEATURES CLIMÁTICAS Y TEMPORALES)
# ==============================================================================
def columnas_features(incluir_clima: bool = True) -> List[str]:
    """Retorna la lista de variables (features) utilizadas en el modelado."""
    if not incluir_clima:
        return ["log_casos", "sem_sin", "sem_cos"]
    return [
        "log_casos",         # Casos en escala logarítmica (estabiliza varianza)
        "lluvia_mm",         # Precipitación acumulada semanal actual
        "temp_max_c",        # Temperatura máxima semanal actual
        "temp_min_c",        # Temperatura mínima semanal actual
        "humedad_rel_pct",   # Humedad relativa semanal actual
        "humedad_lag1",      # Humedad con 1 semana de rezago (supervivencia del mosquito adulto)
        "lluvia_lag3",       # Lluvia con 3 semanas de rezago (ciclo de criaderos y eclosión larvaria)
        "temp_min_lag8",     # Temperatura mínima con 8 semanas de rezago (periodo de incubación extrínseca)
        "temp_max_lag12",    # Temperatura máxima con 12 semanas de rezago (inercia térmica estacional)
        "sem_sin",           # Componente seno de la semana epidemiológica (periodicidad anual)
        "sem_cos",           # Componente coseno de la semana epidemiológica (periodicidad anual)
    ]


def crear_features(df: pd.DataFrame, incluir_clima: bool = True) -> pd.DataFrame:
    """Aplica ingeniería de variables sobre la serie temporal.

    Operaciones realizadas:
    - Transformación logarítmica log(1 + casos): Estabiliza la varianza de brotes.
    - Rezagos climáticos basados en biología del vector Aedes aegypti:
      * Humedad (lag 1): Afecta la longevidad del mosquito adulto a corto plazo.
      * Lluvia (lag 3): Periodo para acumulación de agua y eclosión (~2-3 semanas).
      * Temp. mínima (lag 8) y máxima (lag 12): Transición estacional e incubación viral.
    - Codificación armónica anual (seno y coseno de 2*pi*semana/52).

    Args:
        df: DataFrame original ordenado cronológicamente.
        incluir_clima: Si es False, solo genera variables autorregresivas y de calendario.

    Returns:
        DataFrame con las nuevas columnas calculadas, descartando las primeras 12 semanas iniciales con nulos.
    """
    df_feat = df.copy()

    # 1. Transformación logarítmica de casos
    df_feat["log_casos"] = np.log1p(df_feat["casos_dengue"])

    # 2. Rezagos climáticos
    if incluir_clima:
        df_feat["humedad_lag1"] = df_feat["humedad_rel_pct"].shift(1)
        df_feat["lluvia_lag3"] = df_feat["lluvia_mm"].shift(3)
        df_feat["temp_min_lag8"] = df_feat["temp_min_c"].shift(8)
        df_feat["temp_max_lag12"] = df_feat["temp_max_c"].shift(12)

    # 3. Codificación cíclica de la semana epidemiológica (1 a 52/53)
    semana = df_feat["semana_epi"].astype(float)
    df_feat["sem_sin"] = np.sin(2 * np.pi * semana / 52.0)
    df_feat["sem_cos"] = np.cos(2 * np.pi * semana / 52.0)

    # 4. Descartar únicamente las primeras 12 semanas iniciales (rezago máximo)
    df_feat = df_feat.iloc[12:].reset_index(drop=True)
    return df_feat


# ==============================================================================
# 3. PREPARACIÓN DE TENSORES 3D Y PARTICIÓN TEMPORAL
# ==============================================================================
def preparar_datos(
    df: pd.DataFrame,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    ajustar_con_todo: bool = False,
    incluir_clima: bool = True,
) -> Dict:
    """Escala los datos, genera ventanas deslizantes temporales 3D y define particiones.

    Partición temporal estricta (sin fuga de información / data leakage):
        - Entrenamiento (Train): Semanas objetivo pertenecientes a los años 2014–2021 (~377 ventanas).
        - Validación (Val): Semanas objetivo pertenecientes al año 2022 (49 ventanas).
        - Prueba (Test): Semanas objetivo en 2023–2024 (101 ventanas).

    Dimensiones de salida:
        - X: (n_ventanas, 26, n_features) -> 26 semanas de historia observada.
        - y: (n_ventanas, 4) -> 4 semanas consecutivas futuras a predecir en log(1+casos).

    Args:
        df: DataFrame con los datos originales.
        ventana: Tamaño de la secuencia de entrada (26 semanas).
        horizonte: Tamaño de la secuencia objetivo (4 semanas).
        ajustar_con_todo: Si es True, entrena los escaladores con toda la serie (para el reentrenamiento final).
        incluir_clima: Si es True, incluye las variables climáticas rezagadas.

    Returns:
        Diccionario con tensores X e y, máscaras booleanas de partición, escaladores y metadatos.
    """
    df_feat = crear_features(df, incluir_clima=incluir_clima)
    features = columnas_features(incluir_clima=incluir_clima)

    n = len(df_feat)
    n_ventanas = n - ventana - horizonte + 1
    if n_ventanas <= 0:
        raise ValueError("El DataFrame no cuenta con suficientes observaciones para la ventana y horizonte dados.")

    idx_obj = np.arange(ventana, n - horizonte + 1)
    anios_feat = df_feat["anio"].to_numpy()

    mask_train = np.zeros(n_ventanas, dtype=bool)
    mask_val = np.zeros(n_ventanas, dtype=bool)
    mask_test = np.zeros(n_ventanas, dtype=bool)

    # Asignación de máscaras según el año del objetivo (las ventanas que cruzan particiones se descartan limpiamente)
    for i, idx in enumerate(idx_obj):
        anios_obj = anios_feat[idx : idx + horizonte]
        if np.all(anios_obj <= 2021):
            mask_train[i] = True
        elif np.all(anios_obj == 2022):
            mask_val[i] = True
        elif np.all((anios_obj >= 2023) & (anios_obj <= 2024)):
            mask_test[i] = True

    # Escalado MinMaxScaler en rango [0, 1]
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


# ==============================================================================
# 4. BASELINES DE REFERENCIA Y MÉTRICAS DE ERROR
# ==============================================================================
def casos_reales_ventanas(datos: Dict, mask: np.ndarray) -> np.ndarray:
    """Extrae la matriz de casos reales observados en escala original (casos/semana) para una máscara dada."""
    idx_obj = datos["idx_obj"][mask]
    casos = datos["df_feat"]["casos_dengue"].to_numpy()
    horizonte = datos["horizonte"]

    reales = np.empty((len(idx_obj), horizonte), dtype=np.float64)
    for i, idx in enumerate(idx_obj):
        reales[i] = casos[idx : idx + horizonte]
    return reales


def baseline_estacional(datos: Dict, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula el baseline estacional histórico (casos de la misma semana epidemiológica del año anterior).

    Returns:
        base_est: Matriz de predicciones (N, horizonte).
        ok: Array booleano que indica las ventanas que cuentan con histórico disponible del año anterior.
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
    """Calcula el baseline de persistencia repitiendo el último valor observado durante el horizonte de 4 semanas."""
    idx_obj = datos["idx_obj"][mask]
    casos = datos["df_feat"]["casos_dengue"].to_numpy()
    horizonte = datos["horizonte"]

    base_per = np.empty((len(idx_obj), horizonte), dtype=np.float64)
    for i, idx in enumerate(idx_obj):
        ult_caso = casos[idx - 1]
        base_per[i] = np.full(horizonte, ult_caso)
    return base_per


def rmse(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """Calcula la Raíz del Error Cuadrático Medio (RMSE) en escala real de casos."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """Calcula el Error Absoluto Medio (MAE) en escala real de casos."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(yt - yp)))


# ==============================================================================
# 5. CAPA PERSONALIZADA Y ARQUITECTURAS NEURONALES RECURRENTES
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="VectorIA")
class LastStepExtract(Layer):
    """Capa serializable de Keras para conexión residual en modelos de series temporales.

    Extrae el último paso temporal de la primera variable (log_casos en t=26) y lo replica
    a lo largo del horizonte futuro (4 semanas). Permite que la red neuronal arranque exactamente
    como persistencia (cuando la capa Dense está inicializada en ceros) y aprenda exclusivamente
    el residuo o forzamiento climático y estacional.
    """

    def __init__(self, horizonte: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.horizonte = int(horizonte)

    def call(self, inputs):
        # inputs shape: (batch, timesteps=26, features), con feature 0 = log_casos
        last = inputs[:, -1, 0:1]
        return tf.repeat(last, repeats=self.horizonte, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"horizonte": self.horizonte})
        return config


def construir_simplernn(
    n_features: int,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    unidades: int = 32,
    lr: float = 0.001,
    l2_reg: float = 0.0,
    usar_residual: bool = False,
    seed: int = 42,
) -> tf.keras.Model:
    """Construye y compila el modelo SimpleRNN."""
    fijar_semilla(seed)
    inp = Input(shape=(ventana, n_features))
    x = SimpleRNN(
        unidades,
        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
        recurrent_regularizer=l2(l2_reg) if l2_reg > 0 else None,
    )(inp)

    if usar_residual:
        dense_out = Dense(horizonte, kernel_initializer="zeros", bias_initializer="zeros")(x)
        last_val = LastStepExtract(horizonte=horizonte)(inp)
        out = Add()([dense_out, last_val])
        modelo = Model(inputs=inp, outputs=out)
    else:
        dense_out = Dense(horizonte)(x)
        modelo = Model(inputs=inp, outputs=dense_out)

    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


def construir_lstm(
    n_features: int,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    u1: int = 32,
    u2: Optional[int] = None,
    dropout: float = 0.0,
    lr: float = 0.001,
    l2_reg: float = 0.001,
    usar_residual: bool = True,
    seed: int = 42,
) -> tf.keras.Model:
    """Construye y compila el modelo LSTM con opción de regularización y residuo autorregresivo."""
    fijar_semilla(seed)
    inp = Input(shape=(ventana, n_features))
    if u2 is not None and u2 > 0:
        x = LSTM(
            u1,
            return_sequences=True,
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
            recurrent_regularizer=l2(l2_reg) if l2_reg > 0 else None,
        )(inp)
        if dropout > 0:
            x = Dropout(dropout)(x)
        x = LSTM(
            u2,
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
            recurrent_regularizer=l2(l2_reg) if l2_reg > 0 else None,
        )(x)
    else:
        x = LSTM(
            u1,
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
            recurrent_regularizer=l2(l2_reg) if l2_reg > 0 else None,
        )(inp)

    if dropout > 0 and (u2 is None or u2 == 0):
        x = Dropout(dropout)(x)

    if usar_residual:
        dense_out = Dense(horizonte, kernel_initializer="zeros", bias_initializer="zeros")(x)
        last_val = LastStepExtract(horizonte=horizonte)(inp)
        out = Add()([dense_out, last_val])
        modelo = Model(inputs=inp, outputs=out)
    else:
        dense_out = Dense(horizonte)(x)
        modelo = Model(inputs=inp, outputs=dense_out)

    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


def construir_gru(
    n_features: int,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    unidades: int = 16,
    lr: float = 0.001,
    l2_reg: float = 0.0001,
    usar_residual: bool = True,
    seed: int = 42,
) -> tf.keras.Model:
    """Construye y compila el modelo GRU con regularización y residuo autorregresivo."""
    fijar_semilla(seed)
    inp = Input(shape=(ventana, n_features))
    x = GRU(
        unidades,
        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
        recurrent_regularizer=l2(l2_reg) if l2_reg > 0 else None,
    )(inp)

    if usar_residual:
        dense_out = Dense(horizonte, kernel_initializer="zeros", bias_initializer="zeros")(x)
        last_val = LastStepExtract(horizonte=horizonte)(inp)
        out = Add()([dense_out, last_val])
        modelo = Model(inputs=inp, outputs=out)
    else:
        dense_out = Dense(horizonte)(x)
        modelo = Model(inputs=inp, outputs=dense_out)

    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


CANDIDATOS_CV: Dict[str, Dict] = {
    "A": {
        "nombre": "SimpleRNN_base",
        "descripcion": "SimpleRNN(32), sin residual, sin L2",
        "tipo": "simplernn",
        "unidades": 32,
        "l2_reg": 0.0,
        "usar_residual": False,
        "incluir_clima": True,
    },
    "B": {
        "nombre": "LSTM_profesor",
        "descripcion": "LSTM(64) -> LSTM(32), Dropout 0.2, sin residual, sin L2",
        "tipo": "lstm_profesor",
        "u1": 64,
        "u2": 32,
        "dropout": 0.2,
        "l2_reg": 0.0,
        "usar_residual": False,
        "incluir_clima": True,
    },
    "C": {
        "nombre": "SimpleRNN_res",
        "descripcion": "SimpleRNN(16), residual, L2=1e-4",
        "tipo": "simplernn",
        "unidades": 16,
        "l2_reg": 1e-4,
        "usar_residual": True,
        "incluir_clima": True,
    },
    "D": {
        "nombre": "LSTM_res_16",
        "descripcion": "LSTM(16), residual, L2=1e-4",
        "tipo": "lstm",
        "u1": 16,
        "l2_reg": 1e-4,
        "usar_residual": True,
        "incluir_clima": True,
    },
    "E": {
        "nombre": "LSTM_res_32",
        "descripcion": "LSTM(32), residual, L2=1e-3",
        "tipo": "lstm",
        "u1": 32,
        "l2_reg": 1e-3,
        "usar_residual": True,
        "incluir_clima": True,
    },
    "F": {
        "nombre": "GRU_res_16",
        "descripcion": "GRU(16), residual, L2=1e-4",
        "tipo": "gru",
        "unidades": 16,
        "l2_reg": 1e-4,
        "usar_residual": True,
        "incluir_clima": True,
    },
    "G": {
        "nombre": "GRU_res_16_noclima",
        "descripcion": "GRU(16), residual, L2=1e-4, SIN variables climáticas (Ablación)",
        "tipo": "gru",
        "unidades": 16,
        "l2_reg": 1e-4,
        "usar_residual": True,
        "incluir_clima": False,
    },
}


def construir_modelo_candidato(
    candidato_id: str,
    n_features: Optional[int] = None,
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    seed: int = 42,
) -> tf.keras.Model:
    """Construye un modelo candidato para la validación cruzada."""
    cfg = CANDIDATOS_CV[candidato_id]
    if n_features is None:
        n_features = len(columnas_features(incluir_clima=cfg["incluir_clima"]))

    tipo = cfg["tipo"]
    if tipo == "simplernn":
        return construir_simplernn(
            n_features=n_features,
            ventana=ventana,
            horizonte=horizonte,
            unidades=cfg["unidades"],
            l2_reg=cfg["l2_reg"],
            usar_residual=cfg["usar_residual"],
            seed=seed,
        )
    elif tipo == "lstm":
        return construir_lstm(
            n_features=n_features,
            ventana=ventana,
            horizonte=horizonte,
            u1=cfg["u1"],
            l2_reg=cfg["l2_reg"],
            usar_residual=cfg["usar_residual"],
            seed=seed,
        )
    elif tipo == "lstm_profesor":
        return construir_lstm(
            n_features=n_features,
            ventana=ventana,
            horizonte=horizonte,
            u1=cfg["u1"],
            u2=cfg["u2"],
            dropout=cfg["dropout"],
            l2_reg=cfg["l2_reg"],
            usar_residual=cfg["usar_residual"],
            seed=seed,
        )
    elif tipo == "gru":
        return construir_gru(
            n_features=n_features,
            ventana=ventana,
            horizonte=horizonte,
            unidades=cfg["unidades"],
            l2_reg=cfg["l2_reg"],
            usar_residual=cfg["usar_residual"],
            seed=seed,
        )
    else:
        raise ValueError(f"Tipo de candidato desconocido: {tipo}")


# ==============================================================================
# 6. ENTRENAMIENTO Y VALIDACIÓN CRUZADA EXPANSIVA (CV)
# ==============================================================================
def entrenar(
    modelo: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 300,
    batch_size: int = 16,
    patience: int = 40,
    lr_patience: int = 12,
    start_from_epoch: int = 30,
    lr_factor: float = 0.5,
    min_lr: float = 1e-5,
    verbose: int = 0,
    seed: int = 42,
) -> tf.keras.callbacks.History:
    """Entrena el modelo con callbacks homogéneos de parada temprana y reducción adaptativa de tasa de aprendizaje."""
    fijar_semilla(seed)
    callbacks = []
    val_data = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_data = (X_val, y_val)
        callbacks.extend([
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                start_from_epoch=start_from_epoch,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_lr=min_lr,
                verbose=0,
            ),
        ])

    return modelo.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )


def validacion_cruzada_expansiva(
    df: pd.DataFrame,
    candidatos: Optional[List[str]] = None,
    folds_val: Optional[List[int]] = None,
    semillas: Tuple[int, ...] = (42, 43),
    ventana: int = VENTANA,
    horizonte: int = HORIZONTE,
    epochs: int = 300,
    batch_size: int = 16,
    patience: int = 40,
    lr_patience: int = 12,
    start_from_epoch: int = 30,
    verbose: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta una validación cruzada temporal de ventana expansiva (Expanding Window CV) sobre los años de entrenamiento.

    Folds de validación: 2018, 2019, 2021, 2022.
    Para cada fold:
        - Entrenamiento: Todo el pasado anterior al año de validación.
        - Escaladores: Ajustados exclusivamente con los datos de entrenamiento del fold.
        - Métrica de selección: Skill Score = 1 - RMSE_modelo / RMSE_persistencia.

    Returns:
        df_detalle: Resultados detallados por candidato, fold y semilla.
        df_resumen: Tabla agregada con media y desviación estándar de Skill, RMSE y MAE.
    """
    if candidatos is None:
        candidatos = list(CANDIDATOS_CV.keys())
    if folds_val is None:
        folds_val = [2018, 2019, 2021, 2022]

    registros = []

    # Pre-generar features completas y sin clima
    df_feat_clima = crear_features(df, incluir_clima=True)
    df_feat_noclima = crear_features(df, incluir_clima=False)

    for cand_id in candidatos:
        cfg = CANDIDATOS_CV[cand_id]
        df_feat = df_feat_clima if cfg["incluir_clima"] else df_feat_noclima
        features = columnas_features(incluir_clima=cfg["incluir_clima"])

        n = len(df_feat)
        n_ventanas = n - ventana - horizonte + 1
        idx_obj = np.arange(ventana, n - horizonte + 1)
        anios_feat = df_feat["anio"].to_numpy()
        casos_reales_arr = df_feat["casos_dengue"].to_numpy()

        for fold_year in folds_val:
            mask_tr = np.array([np.all(anios_feat[idx : idx + horizonte] < fold_year) for idx in idx_obj])
            mask_va = np.array([np.all(anios_feat[idx : idx + horizonte] == fold_year) for idx in idx_obj])

            esc_x = MinMaxScaler(feature_range=(0, 1))
            esc_y = MinMaxScaler(feature_range=(0, 1))

            filtro_tr = df_feat["anio"] < fold_year
            esc_x.fit(df_feat.loc[filtro_tr, features])
            esc_y.fit(df_feat.loc[filtro_tr, ["log_casos"]])

            X_mat = esc_x.transform(df_feat[features])
            y_vec = esc_y.transform(df_feat[["log_casos"]]).flatten()

            X_all = np.empty((n_ventanas, ventana, len(features)), dtype=np.float32)
            y_all = np.empty((n_ventanas, horizonte), dtype=np.float32)
            for i in range(n_ventanas):
                X_all[i] = X_mat[i : i + ventana]
                y_all[i] = y_vec[i + ventana : i + ventana + horizonte]

            X_tr, y_tr = X_all[mask_tr], y_all[mask_tr]
            X_va, y_va = X_all[mask_va], y_all[mask_va]

            idx_val_indices = idx_obj[mask_va]
            y_real = np.array([casos_reales_arr[i : i + horizonte] for i in idx_val_indices], dtype=np.float64)
            y_per = np.array([np.full(horizonte, casos_reales_arr[i - 1]) for i in idx_val_indices], dtype=np.float64)
            rmse_persistencia = rmse(y_real, y_per)

            for seed in semillas:
                modelo = construir_modelo_candidato(cand_id, n_features=len(features), ventana=ventana, horizonte=horizonte, seed=seed)
                hist = entrenar(
                    modelo,
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    lr_patience=lr_patience,
                    start_from_epoch=start_from_epoch,
                    verbose=verbose,
                    seed=seed,
                )

                pred_sc = modelo.predict(X_va, verbose=0)
                pred_log = esc_y.inverse_transform(pred_sc)
                pred_casos = np.clip(np.expm1(pred_log), 0.0, None)

                r_val = rmse(y_real, pred_casos)
                m_val = mae(y_real, pred_casos)
                sk_val = 1.0 - (r_val / rmse_persistencia)

                registros.append({
                    "candidato_id": cand_id,
                    "candidato_nombre": cfg["nombre"],
                    "descripcion": cfg["descripcion"],
                    "fold": fold_year,
                    "semilla": seed,
                    "epocas_entrenadas": len(hist.history["loss"]),
                    "RMSE_persistencia": rmse_persistencia,
                    "RMSE": r_val,
                    "MAE": m_val,
                    "Skill": sk_val,
                })

    df_detalle = pd.DataFrame(registros)

    # Resumen por candidato (promediando semillas por fold, luego promediando folds)
    df_por_fold = df_detalle.groupby(["candidato_id", "candidato_nombre", "descripcion", "fold"])[["RMSE", "MAE", "Skill"]].mean().reset_index()
    resumen_filas = []
    for cand_id, grupo in df_por_fold.groupby("candidato_id"):
        cand_nombre = grupo["candidato_nombre"].iloc[0]
        desc = grupo["descripcion"].iloc[0]
        resumen_filas.append({
            "Candidato": cand_id,
            "Modelo": cand_nombre,
            "Descripción": desc,
            "Skill_mean": grupo["Skill"].mean(),
            "Skill_std": grupo["Skill"].std(),
            "RMSE_mean": grupo["RMSE"].mean(),
            "RMSE_std": grupo["RMSE"].std(),
            "MAE_mean": grupo["MAE"].mean(),
            "MAE_std": grupo["MAE"].std(),
        })

    df_resumen = pd.DataFrame(resumen_filas).sort_values("Skill_mean", ascending=False).reset_index(drop=True)
    return df_detalle, df_resumen


def evaluar_en(datos: Dict, modelo: tf.keras.Model, particion: str) -> Dict:
    """Evalúa un modelo sobre una partición temporal ('val' o 'test') en escala real de casos."""
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


# ==============================================================================
# 7. PERSISTENCIA DE ARTEFACTOS Y API DE INTEGRACIÓN (STREAMLIT)
# ==============================================================================
def guardar_artefactos(
    modelo: tf.keras.Model,
    esc_x: MinMaxScaler,
    esc_y: MinMaxScaler,
    nombre: str,
    metadata: Optional[Dict] = None,
    ruta: Optional[Union[str, Path]] = None,
) -> None:
    """Guarda el modelo entrenado (.keras) y un único archivo de escaladores (_scalers.joblib).

    Args:
        modelo: Modelo Keras serializable.
        esc_x: Escalador MinMaxScaler de las variables de entrada.
        esc_y: Escalador MinMaxScaler de la variable objetivo.
        nombre: Prefijo o identificador del modelo (ej: 'lstm_dengue_final').
        metadata: Parámetro opcional para compatibilidad.
        ruta: Directorio de destino (por defecto 'models/LSTM/').
    """
    ruta_dir = Path(ruta) if ruta is not None else RUTA_MODELOS
    ruta_dir.mkdir(parents=True, exist_ok=True)

    modelo.save(ruta_dir / f"{nombre}.keras")
    joblib.dump({"esc_x": esc_x, "esc_y": esc_y}, ruta_dir / f"{nombre}_scalers.joblib")


def cargar_artefactos(
    nombre: str,
    ruta: Optional[Union[str, Path]] = None,
) -> Tuple[tf.keras.Model, MinMaxScaler, MinMaxScaler]:
    """Carga el modelo (.keras) y sus escaladores (_scalers.joblib) desde disco.

    Args:
        nombre: Prefijo o identificador del modelo (ej: 'lstm_dengue_final').
        ruta: Directorio de almacenamiento (por defecto 'models/LSTM/').

    Returns:
        Tupla (modelo, esc_x, esc_y).
    """
    ruta_dir = Path(ruta) if ruta is not None else RUTA_MODELOS
    ruta_modelo = ruta_dir / f"{nombre}.keras"
    ruta_scalers = ruta_dir / f"{nombre}_scalers.joblib"

    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")
    if not ruta_scalers.exists():
        raise FileNotFoundError(f"No se encontró el archivo de escaladores en: {ruta_scalers}")

    modelo = load_model(ruta_modelo, compile=False)
    scalers = joblib.load(ruta_scalers)
    esc_x = scalers["esc_x"]
    esc_y = scalers["esc_y"]
    return modelo, esc_x, esc_y


def predecir_4_semanas(
    modelo: tf.keras.Model,
    hist_df: pd.DataFrame,
    esc_x: MinMaxScaler,
    esc_y: MinMaxScaler,
    ventana: int = VENTANA,
) -> np.ndarray:
    """API de inferencia para la aplicación Streamlit: predice las próximas 4 semanas epidemiológicas.

    Args:
        modelo: Red recurrente cargada.
        hist_df: DataFrame con la historia reciente observada (mínimo 38 semanas para calcular rezagos y ventana).
        esc_x: Escalador de features de entrada.
        esc_y: Escalador de la variable objetivo.
        ventana: Tamaño de la secuencia de entrada (26 semanas).

    Returns:
        np.ndarray de 4 elementos con los casos pronosticados en escala real para t+1, t+2, t+3, t+4.
    """
    # Identificar si el escalador fue ajustado con o sin clima
    if hasattr(esc_x, "feature_names_in_"):
        features = list(esc_x.feature_names_in_)
        incluir_clima = len(features) > 3
    else:
        features = columnas_features(incluir_clima=True)
        incluir_clima = True

    sub_feat = crear_features(hist_df, incluir_clima=incluir_clima)
    if len(sub_feat) < ventana:
        raise ValueError(f"hist_df debe contener suficientes semanas tras rezagos ({len(sub_feat)} < {ventana}).")

    ventana_df = sub_feat.iloc[-ventana:][features]
    ventana_scaled = esc_x.transform(ventana_df).reshape(1, ventana, len(features))

    pred_scaled = modelo.predict(ventana_scaled, verbose=0)
    pred_log = esc_y.inverse_transform(pred_scaled)
    pred_casos = np.expm1(pred_log)
    pred_casos = np.clip(pred_casos, 0.0, None)
    return pred_casos[0].astype(np.float64)


# ==============================================================================
# 8. PRONÓSTICO RECURSIVO HACIA EL FUTURO (2025–2026)
# ==============================================================================
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
    """Genera pronósticos recursivos semana a semana hasta finalizar el año epidemiológico indicado (2026).

    Metodología:
    1. Genera el calendario de fechas futuras iniciando en la semana inmediatamente posterior al dataset.
    2. Utiliza la función `anio_semana_epi` para garantizar numeración exacta y detenerse cuando el año supere 2026.
    3. Asigna la climatología histórica promedio semanal como forzamiento esperado para 2025 y 2026.
    4. En cada paso recursivo, pronostica el horizonte y retroalimenta la predicción t+1 en la serie histórica.

    Returns:
        pd.DataFrame con exactamente 105 semanas de pronóstico (52 de 2025 y 53 de 2026).
    """
    df_base = df.sort_values("fecha_inicio_semana").reset_index(drop=True).copy()
    ult_fecha = df_base["fecha_inicio_semana"].max()

    # Climatología histórica promedio agrupada por semana epidemiológica
    climatologia = (
        df_base.groupby("semana_epi")[COLUMNAS_CLIMA]
        .mean()
        .to_dict(orient="index")
    )

    # Generación rigurosa del calendario de semanas futuras usando anio_semana_epi
    fechas_futuras = []
    anios_futuros = []
    semanas_futuras = []

    fecha_cursor = ult_fecha + pd.Timedelta(days=7)
    while True:
        a_epi, s_epi = anio_semana_epi(fecha_cursor)
        if a_epi > hasta_anio:
            break
        fechas_futuras.append(fecha_cursor)
        anios_futuros.append(a_epi)
        semanas_futuras.append(s_epi)
        fecha_cursor += pd.Timedelta(days=7)

    df_futuras = pd.DataFrame({
        "fecha_inicio_semana": fechas_futuras,
        "anio": anios_futuros,
        "semana_epi": semanas_futuras,
    })

    # Asignar variables climáticas futuras
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

    if hasattr(esc_x, "feature_names_in_"):
        features = list(esc_x.feature_names_in_)
        incluir_clima = len(features) > 3
    else:
        features = columnas_features(incluir_clima=True)
        incluir_clima = True

    # Bucle de inferencia recursiva semana a semana
    for idx_fut in range(idx_inicio_futuro, len(df_extendido)):
        sub_df = df_extendido.iloc[:idx_fut].copy()
        sub_feat = crear_features(sub_df, incluir_clima=incluir_clima)
        if len(sub_feat) < ventana:
            continue

        ventana_df = sub_feat.iloc[-ventana:][features]
        ventana_scaled = esc_x.transform(ventana_df).reshape(1, ventana, len(features))

        pred_scaled = modelo.predict(ventana_scaled, verbose=0)
        pred_scaled = np.clip(pred_scaled, -1.0, 5.0)
        pred_log = esc_y.inverse_transform(pred_scaled)
        pred_log = np.clip(pred_log, 0.0, np.log(100000.0))
        pred_casos = np.expm1(pred_log)
        pred_casos = np.clip(pred_casos, 0.0, 50000.0)

        caso_siguiente = float(np.nan_to_num(pred_casos[0, 0], nan=0.0, posinf=5000.0, neginf=0.0))
        df_extendido.loc[idx_fut, "casos_dengue"] = caso_siguiente

    df_resultado = df_extendido.iloc[idx_inicio_futuro:].copy().reset_index(drop=True)
    df_resultado["casos_pred"] = df_resultado["casos_dengue"]
    return df_resultado


# ==============================================================================
# 9. CONSOLIDACIÓN DE TABLAS DE EVALUACIÓN Y PRONÓSTICO
# ==============================================================================
def tabla_evaluacion(
    datos: Dict,
    preds_dict: Dict[str, np.ndarray],
    modelo_principal_nombre: str = "LSTM",
) -> pd.DataFrame:
    """Construye el DataFrame consolidado de evaluación sobre el conjunto de prueba (2023–2024)."""
    idx_test = datos["idx_obj"][datos["masks"]["test"]]
    df_feat = datos["df_feat"]

    pred_rnn = preds_dict["SimpleRNN"][:, 0]
    pred_lstm = preds_dict["LSTM"][:, 0]
    casos_reales = df_feat["casos_dengue"].iloc[idx_test].to_numpy()

    df_eval = pd.DataFrame({
        "fecha_inicio_semana": df_feat["fecha_inicio_semana"].iloc[idx_test].to_numpy(),
        "anio": df_feat["anio"].iloc[idx_test].to_numpy(),
        "semana_epi": df_feat["semana_epi"].iloc[idx_test].to_numpy(),
        "casos_reales": casos_reales,
        "pred_SimpleRNN": np.round(pred_rnn, 1),
        "pred_LSTM": np.round(pred_lstm, 1),
        "periodo": "evaluacion_test",
        "clima_usado": "observado_real",
        "modelo_principal": modelo_principal_nombre,
    })

    # Alerta acumulada a 4 semanas (suma de t, t+1, t+2, t+3); NaN en las últimas 3 donde no hay 4 semanas
    df_eval["casos_acum_4sem_LSTM"] = np.round(df_eval["pred_LSTM"].rolling(4).sum().shift(-3), 1)
    return df_eval
