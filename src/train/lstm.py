"""Módulo de entrenamiento, evaluación y pronóstico con modelos recurrentes (SimpleRNN y LSTM)
para la predicción de casos de dengue en Costa Rica (Proyecto VectorIA).

Este módulo proporciona el flujo de trabajo completo:
1. Carga y ordenamiento cronológico del dataset nacional.
2. Ingeniería de variables: transformación logarítmica, rezagos climáticos biológicos
   y codificación cíclica (seno/coseno) de la semana epidemiológica.
3. Construcción de tensores 3D con ventanas deslizantes temporales (Lookback = 26 semanas, Horizonte = 4 semanas).
4. Partición temporal estricta sin fuga de información (Train <= 2021, Val = 2022, Test = 2023-2024).
5. Cálculo de baselines de referencia (Estacional histórico y Persistencia).
6. Construcción y entrenamiento de redes neuronales recurrentes (SimpleRNN y LSTM optimizada con conexión residual).
7. Evaluación en escala real de casos (RMSE y MAE multihorizonte).
8. Persistencia de artefactos (.keras, .joblib).
9. Pronóstico recursivo hacia el futuro (2025–2026) con climatología estacional.
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
from tensorflow.keras.layers import Add, Dense, Dropout, Input, Layer, LSTM, SimpleRNN
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
# 1. CARGA Y LECTURA DE DATOS
# ==============================================================================
def cargar_datos(ruta: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Carga el dataset de casos de dengue y variables climáticas nacionales.

    Realiza una búsqueda inteligente del archivo 'vectoria_dengue_nacional_real.csv'
    en diversas rutas estándar del proyecto o en la carpeta Downloads del usuario.
    Ordena cronológicamente los registros por 'fecha_inicio_semana'.

    Args:
        ruta: Ruta opcional personalizada hacia el archivo CSV.

    Returns:
        pd.DataFrame con los datos limpios y ordenados en orden cronológico ascendente.
    """
    if ruta is None:
        # Rutas candidatas habituales dentro del repositorio y entornos de desarrollo
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

        # Si no está en las rutas relativas directas, busca en el árbol del proyecto o Downloads
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

    # Lectura del CSV parseando fechas y ordenando cronológicamente
    df = pd.read_csv(ruta_csv, parse_dates=["fecha_inicio_semana"])
    df = df.sort_values("fecha_inicio_semana").reset_index(drop=True)
    return df


# ==============================================================================
# 2. INGENIERÍA DE VARIABLES (FEATURES CLIMÁTICAS Y TEMPORALES)
# ==============================================================================
def columnas_features() -> List[str]:
    """Retorna la lista de variables (features) utilizadas en el entrenamiento de los modelos."""
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


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica ingeniería de variables sobre la serie temporal.

    Operaciones realizadas:
    - Transformación logarítmica log(1 + casos): Convierte distribuciones asimétricas
      con brotes en distribuciones más gaussianas para optimizadores basados en gradiente.
    - Rezagos climáticos basados en biología del vector Aedes aegypti:
      * Humedad (lag 1): Afecta la longevidad del mosquito hembra a corto plazo.
      * Lluvia (lag 3): Periodo requerido para acumulación de agua y desarrollo de larvas a adultos (~2-3 semanas).
      * Temp. mínima (lag 8) y máxima (lag 12): Marcan transiciones estacionales y periodos de incubación viral.
    - Codificación cíclica (seno y coseno de 2*pi*semana/52):
      Permite al modelo entender que la semana 52 y la semana 1 están temporalmente contiguas.

    Args:
        df: DataFrame original ordenado cronológicamente.

    Returns:
        DataFrame con las nuevas columnas calculadas, descartando las primeras 12 semanas iniciales con nulos.
    """
    df_feat = df.copy()

    # 1. Transformación logarítmica de casos
    df_feat["log_casos"] = np.log1p(df_feat["casos_dengue"])

    # 2. Rezagos climáticos
    df_feat["humedad_lag1"] = df_feat["humedad_rel_pct"].shift(1)
    df_feat["lluvia_lag3"] = df_feat["lluvia_mm"].shift(3)
    df_feat["temp_min_lag8"] = df_feat["temp_min_c"].shift(8)
    df_feat["temp_max_lag12"] = df_feat["temp_max_c"].shift(12)

    # 3. Codificación cíclica de la semana epidemiológica (1 a 52/53)
    semana = df_feat["semana_epi"].astype(float)
    df_feat["sem_sin"] = np.sin(2 * np.pi * semana / 52.0)
    df_feat["sem_cos"] = np.cos(2 * np.pi * semana / 52.0)

    # 4. Descartar únicamente las primeras 12 semanas iniciales (rezago máximo)
    features = columnas_features()
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
) -> Dict:
    """Escala los datos, genera ventanas deslizantes temporales 3D y define particiones.

    Partición temporal estricta (sin fuga de información / data leakage):
        - Entrenamiento (Train): Semanas objetivo pertenecientes a los años 2014–2021 (~390 ventanas).
        - Validación (Val): Semanas objetivo pertenecientes al año 2022 (52 ventanas).
        - Prueba (Test): Semanas objetivo en 2023–2024 (años de brote epidémico, ~100 ventanas).

    Dimensiones de salida:
        - X: (n_ventanas, 26, 11) -> 26 semanas de historia con 11 variables por semana.
        - y: (n_ventanas, 4) -> 4 semanas consecutivas futuras a predecir en log(1+casos).

    Args:
        df: DataFrame con los datos originales.
        ventana: Tamaño de la secuencia de entrada (26 semanas).
        horizonte: Tamaño de la secuencia objetivo (4 semanas).
        ajustar_con_todo: Si es True, entrena los escaladores con toda la serie (para el reentrenamiento final).

    Returns:
        Diccionario con tensores X e y, máscaras booleanas de partición, escaladores y metadatos.
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
        # Ajuste exclusivo con datos de entrenamiento para evitar fuga de información
        filtro_train = df_feat["anio"] <= 2021
        esc_x.fit(df_feat.loc[filtro_train, features])
        esc_y.fit(df_feat.loc[filtro_train, ["log_casos"]])

    X_mat = esc_x.transform(df_feat[features])
    y_vec = esc_y.transform(df_feat[["log_casos"]]).flatten()

    # Construcción de tensores 3D (X) y 2D (y)
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
# 5. ARQUITECTURAS NEURONALES RECURRENTES (SimpleRNN Y LSTM)
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="VectorIA")
class LastStepExtract(Layer):
    """Capa serializable de Keras para conexión residual en modelos de series temporales.

    Extrae el último paso temporal de la primera variable (log_casos en t=26) y lo replica
    a lo largo del horizonte futuro (4 semanas). Permite que la red neuronal aprenda solo
    el residuo o cambio de tendencia inducido por el clima y estacionalidad.
    """

    def __init__(self, horizonte: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.horizonte = horizonte

    def call(self, inputs):
        # inputs shape: (batch, timesteps=26, features=11), con feature 0 = log_casos
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
) -> tf.keras.Model:
    """Construye y compila el modelo base SimpleRNN:
    Input(shape=(26, n_features)) -> SimpleRNN(32) -> Dense(4).
    """
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
    u1: int = 32,
    u2: Optional[int] = None,
    dropout: float = 0.2,
    lr: float = 0.001,
    l2_reg: float = 0.001,
    usar_residual: bool = True,
) -> tf.keras.Model:
    """Construye y compila el modelo LSTM optimizado con regularización L2 y conexión residual.

    Diseño arquitectónico:
    1. Celda LSTM con regularización de pesos (L2) para evitar sobreajuste en series cortas.
    2. Capa Dense lineal para estimar la delta o cambio futuro.
    3. Conexión residual directa con el último valor de casos conocido, logrando
       un pronóstico autoregresivo suave y reduciendo significativamente el RMSE/MAE.
    """
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

    if dropout > 0:
        x = Dropout(dropout)(x)

    dense_out = Dense(horizonte)(x)

    if usar_residual:
        last_val = LastStepExtract(horizonte=horizonte)(inp)
        out = Add()([dense_out, last_val])
        modelo = Model(inputs=inp, outputs=out)
    else:
        modelo = Model(inputs=inp, outputs=dense_out)

    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


# ==============================================================================
# 6. ENTRENAMIENTO Y EVALUACIÓN
# ==============================================================================
def entrenar(
    modelo: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 25,
    verbose: int = 0,
) -> tf.keras.callbacks.History:
    """Entrena el modelo neuronal aplicando parada temprana (EarlyStopping) si se provee conjunto de validación.

    Args:
        modelo: Modelo de Keras compilado.
        X_train, y_train: Tensores de entrenamiento.
        X_val, y_val: Tensores opcionales de validación.
        epochs: Límite máximo de épocas.
        batch_size: Tamaño de lote.
        patience: Número de épocas sin mejora en val_loss antes de detener y restaurar mejores pesos.
        verbose: Nivel de verbosidad (0 silencioso).

    Returns:
        Objeto History con el registro de pérdida por época.
    """
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
    """Evalúa un modelo sobre una partición temporal ('val' o 'test') en escala real de casos.

    Pasos:
    1. Predicción en escala normalizada [0, 1].
    2. Inversión del MinMaxScaler con inverse_transform.
    3. Inversión de la transformación logarítmica con expm1 (exp(y) - 1).
    4. Truncado a cero (los casos de dengue no pueden ser negativos).
    5. Cálculo de métricas globales (RMSE, MAE) y específicas para cada semana futura (h=1..4).

    Returns:
        Diccionario con predicciones en casos reales, valores reales y métricas de error.
    """
    mask = datos["masks"][particion]
    X_eval = datos["X"][mask]
    y_real = casos_reales_ventanas(datos, mask)

    # Inferencia y desescalado
    y_pred_scaled = modelo.predict(X_eval, verbose=0)
    y_pred_log = datos["esc_y"].inverse_transform(y_pred_scaled)
    y_pred_casos = np.expm1(y_pred_log)
    y_pred_casos = np.clip(y_pred_casos, 0, None)

    # Métricas agregadas y por horizonte
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
# 7. PERSISTENCIA DE ARTEFACTOS Y METADATOS
# ==============================================================================
def guardar_artefactos(
    modelo: tf.keras.Model,
    esc_x: MinMaxScaler,
    esc_y: MinMaxScaler,
    nombre: str,
    metadata: Optional[Dict] = None,
    ruta: Optional[Union[str, Path]] = None,
) -> None:
    """Guarda el modelo entrenado (.keras) y los escaladores (.joblib).

    Args:
        modelo: Modelo Keras a serializar.
        esc_x: Escalador de variables de entrada.
        esc_y: Escalador de la variable objetivo.
        nombre: Prefijo o identificador del modelo (ej: 'lstm_dengue').
        metadata: Parámetro opcional mantenido por compatibilidad (no se genera .json).
        ruta: Directorio de destino (por defecto 'models/LSTM/').
    """
    if ruta is None:
        ruta = RUTA_MODELOS
    ruta_dir = Path(ruta)
    ruta_dir.mkdir(parents=True, exist_ok=True)

    modelo.save(ruta_dir / f"{nombre}.keras")
    joblib.dump(esc_x, ruta_dir / f"{nombre}_esc_x.joblib")
    joblib.dump(esc_y, ruta_dir / f"{nombre}_esc_y.joblib")


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
    """Genera pronósticos recursivos semana a semana hasta el final del año indicado (ej. 2026).

    Metodología:
    1. Si no se dispone de pronóstico meteorológico oficial futuro, se utiliza la climatología
       histórica promedio para cada semana epidemiológica (1 a 52).
    2. En cada paso temporal futuro, se construye la ventana de las últimas 26 semanas,
       se realiza la inferencia con el modelo y el caso predicho para la semana t+1 se
       retroalimenta en la serie para permitir pronósticos de largo plazo continuos.

    Returns:
        pd.DataFrame con las semanas futuras proyectadas y los casos estimados.
    """
    df_base = df.sort_values("fecha_inicio_semana").reset_index(drop=True).copy()
    ult_fecha = df_base["fecha_inicio_semana"].max()

    # Climatología histórica promedio agrupada por semana epidemiológica
    climatologia = (
        df_base.groupby("semana_epi")[COLUMNAS_CLIMA]
        .mean()
        .to_dict(orient="index")
    )

    # Generación del calendario de semanas futuras
    fechas_futuras = []
    fecha_cursor = ult_fecha + pd.Timedelta(days=7)
    while True:
        anio_cursor = fecha_cursor.year
        if anio_cursor > hasta_anio:
            break
        fechas_futuras.append(fecha_cursor)
        fecha_cursor += pd.Timedelta(days=7)

    df_futuras = pd.DataFrame({"fecha_inicio_semana": fechas_futuras})
    df_futuras["anio"] = df_futuras["fecha_inicio_semana"].dt.isocalendar().year
    df_futuras["semana_epi"] = df_futuras["fecha_inicio_semana"].dt.isocalendar().week

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

    features = columnas_features()

    # Bucle de inferencia recursiva semana a semana
    for idx_fut in range(idx_inicio_futuro, len(df_extendido)):
        # Tomar todas las filas hasta la anterior que ya tienen casos observados o predichos
        sub_df = df_extendido.iloc[:idx_fut].copy()
        sub_feat = crear_features(sub_df)
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
# 9. CONSOLIDACIÓN DE TABLAS DE EVALUACIÓN
# ==============================================================================
def tabla_evaluacion(datos: Dict, preds_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Construye el DataFrame consolidado de evaluación sobre el conjunto de prueba (horizonte 1 semana)."""
    idx_test = datos["idx_obj"][datos["masks"]["test"]]
    df_feat = datos["df_feat"]

    pred_rnn = preds_dict["SimpleRNN"][:, 0]
    pred_lstm = preds_dict["LSTM"][:, 0]
    casos_reales = df_feat["casos_dengue"].iloc[idx_test].to_numpy()

    filas = {
        "fecha_inicio_semana": df_feat["fecha_inicio_semana"].iloc[idx_test].to_numpy(),
        "anio": df_feat["anio"].iloc[idx_test].to_numpy(),
        "semana_epi": df_feat["semana_epi"].iloc[idx_test].to_numpy(),
        "casos_observados": casos_reales,
        "pred_SimpleRNN": pred_rnn,
        "pred_LSTM": pred_lstm,
        "casos_estimados": pred_lstm,
        "periodo": "evaluacion_" + df_feat["anio"].iloc[idx_test].astype(str).to_numpy(),
        "clima_usado": "observado_real",
    }
    df_eval = pd.DataFrame(filas)
    df_eval["casos_acum_4sem_LSTM"] = df_eval["pred_LSTM"].rolling(4, min_periods=1).sum().shift(-3)
    df_eval["casos_acum_4sem_LSTM"] = df_eval["casos_acum_4sem_LSTM"].bfill().ffill()
    return df_eval
