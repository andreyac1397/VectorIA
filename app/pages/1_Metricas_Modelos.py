"""Panel de métricas de VectorIA.

Ubicación: app/pages/3_Metricas_Modelos.py
Los datos se leen desde data/processed y no se inventan métricas ausentes.
"""
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.theme import aplicar_tema, mostrar_identidad, mostrar_encabezado

DATA = ROOT / "data" / "processed"
CNN = DATA / "cnn"

st.set_page_config(page_title="Métricas | VectorIA", page_icon="📊", layout="wide")
aplicar_tema()
mostrar_identidad()
mostrar_encabezado(
    "Métricas de los modelos",
    "Evaluación y comparación de los modelos CNN, LSTM y ANN de VectorIA.",
)
st.caption("Se muestran resultados de archivos CSV existentes. Las métricas de validación y prueba no son intercambiables.")
st.divider()


@st.cache_data(show_spinner=False)
def leer_csv(ruta: str, mtime: int) -> pd.DataFrame:
    del mtime
    return pd.read_csv(ruta, encoding="utf-8-sig")


def cargar(ruta: Path) -> pd.DataFrame | None:
    if not ruta.is_file():
        return None
    try:
        return leer_csv(str(ruta), ruta.stat().st_mtime_ns)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        st.warning(f"No se pudo leer {ruta.name}: {exc}")
        return None


def numero(valor):
    try:
        return float(valor)
    except (ValueError, TypeError):
        return None


def porcentaje(valor):
    n = numero(valor)
    return f"{n:.2%}" if n is not None else "—"


def decimal(valor, digitos=2):
    n = numero(valor)
    return f"{n:,.{digitos}f}" if n is not None else "—"


def tabla(df: pd.DataFrame):
    st.dataframe(df, hide_index=True, width="stretch")


def grafico_barras(df: pd.DataFrame, eje_x: str, eje_y: str, titulo: str, ordenar=None):
    if df.empty:
        return
    grafico = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{eje_x}:N", sort=ordenar, title=None),
        y=alt.Y(f"{eje_y}:Q", title=eje_y),
        tooltip=[alt.Tooltip(f"{eje_x}:N"), alt.Tooltip(f"{eje_y}:Q", format=".3f")],
    ).properties(height=300, title=titulo)
    st.altair_chart(grafico, width="stretch")


def matriz_confusion(df: pd.DataFrame, titulo: str):
    if df is None or df.empty:
        return
    matriz = df.copy()
    # El CSV ANN fue guardado con índice: primera columna sin nombre.
    primera = matriz.columns[0]
    matriz = matriz.rename(columns={primera: "Clase real"})
    matriz["Clase real"] = matriz["Clase real"].astype(str)
    columnas = [c for c in matriz.columns if c != "Clase real"]
    if not columnas:
        tabla(matriz)
        return
    long = matriz.melt(id_vars="Clase real", value_vars=columnas,
                       var_name="Clase predicha", value_name="Cantidad")
    long["Cantidad"] = pd.to_numeric(long["Cantidad"], errors="coerce")
    grafico = alt.Chart(long).mark_rect().encode(
        x=alt.X("Clase predicha:N", title="Predicción"),
        y=alt.Y("Clase real:N", title="Valor real"),
        color=alt.Color("Cantidad:Q"),
        tooltip=["Clase real:N", "Clase predicha:N", "Cantidad:Q"],
    ).properties(height=320, title=titulo)
    etiquetas = alt.Chart(long).mark_text().encode(
        x="Clase predicha:N", y="Clase real:N",
        text=alt.Text("Cantidad:Q", format=".0f"),
        color=alt.condition("datum.Cantidad > 0", alt.value("white"), alt.value("black")),
    )
    st.altair_chart(grafico + etiquetas, width="stretch")


def grafico_historial(df: pd.DataFrame):
    requeridas = {"epoca", "accuracy", "val_accuracy", "loss", "val_loss"}
    if not requeridas.issubset(df.columns):
        st.info("El historial no contiene las columnas esperadas para graficar.")
        tabla(df)
        return
    for variables, titulo in [(["accuracy", "val_accuracy"], "Accuracy por época"),
                               (["loss", "val_loss"], "Pérdida por época")]:
        largo = df[["epoca", *variables]].melt("epoca", var_name="Serie", value_name="Valor")
        largo["epoca"] = pd.to_numeric(largo["epoca"], errors="coerce")
        largo["Valor"] = pd.to_numeric(largo["Valor"], errors="coerce")
        chart = alt.Chart(largo).mark_line(point=False).encode(
            x=alt.X("epoca:Q", title="Época"),
            y=alt.Y("Valor:Q", title="Accuracy" if "Accuracy" in titulo else "Pérdida"),
            color="Serie:N",
            tooltip=["epoca:Q", "Serie:N", alt.Tooltip("Valor:Q", format=".4f")],
        ).properties(height=300, title=titulo)
        st.altair_chart(chart, width="stretch")


tab_cnn, tab_lstm, tab_ann = st.tabs([
    "🔬 CNN y MobileNetV2", "📈 LSTM y comparativas", "🛡️ ANN"
])

with tab_cnn:
    st.subheader("Identificación de especies de mosquitos")
    cnn_metrics = cargar(CNN / "cnn_validation_metrics.csv")
    # Archivo futuro: debe contener las mismas columnas que cnn_validation_metrics.csv.
    mobilenet_path = CNN / "mobilenetv2_validation_metrics.csv"
    mobile_metrics = cargar(mobilenet_path)
    dfs = [d for d in (cnn_metrics, mobile_metrics) if d is not None and not d.empty]
    if dfs:
        resumen_cnn = pd.concat(dfs, ignore_index=True)
        columnas = ["Modelo", "Accuracy_validation", "Precision_macro_validation",
                    "Recall_macro_validation", "F1_macro_validation", "Mejor_val_loss",
                    "Epoca_mejor_val_loss", "Epocas_ejecutadas"]
        existentes = [c for c in columnas if c in resumen_cnn.columns]
        tabla(resumen_cnn[existentes])
        metricas = ["Accuracy_validation", "Precision_macro_validation",
                    "Recall_macro_validation", "F1_macro_validation"]
        disponibles = [c for c in metricas if c in resumen_cnn.columns]
        if "Modelo" in resumen_cnn and disponibles:
            largo = resumen_cnn[["Modelo", *disponibles]].melt(
                id_vars="Modelo", var_name="Métrica", value_name="Valor")
            largo["Valor"] = pd.to_numeric(largo["Valor"], errors="coerce")
            grafico = alt.Chart(largo).mark_bar().encode(
                x=alt.X("Métrica:N", title=None),
                y=alt.Y("Valor:Q", scale=alt.Scale(domain=[0, 1]), title="Puntuación"),
                color="Modelo:N", xOffset="Modelo:N",
                tooltip=["Modelo:N", "Métrica:N", alt.Tooltip("Valor:Q", format=".3f")],
            ).properties(height=330, title="Métricas de validación por modelo")
            st.altair_chart(grafico, width="stretch")
    else:
        st.info("Todavía no hay métricas de CNN disponibles.")

    if mobile_metrics is None:
        st.info(
            "**MobileNetV2: métricas pendientes.** Cuando Mau entregue el CSV, guárdenlo "
            "en `data/processed/cnn/mobilenetv2_validation_metrics.csv` con las columnas "
            "de `cnn_validation_metrics.csv`. Esta página lo leerá automáticamente al actualizarse. "
            "Para comparar ambos modelos, deben evaluarse con el mismo conjunto de imágenes."
        )
    else:
        st.success("Se encontró el archivo de métricas de MobileNetV2.")

    st.markdown("#### Matriz de confusión — CNN propia (validación)")
    predicciones = cargar(CNN / "cnn_validation_predictions.csv")
    if predicciones is not None and {"class_name", "pred_class_name"}.issubset(predicciones.columns):
        clases = list(dict.fromkeys(
            list(predicciones["class_name"].dropna()) +
            list(predicciones["pred_class_name"].dropna())
        ))
        confusion = pd.crosstab(
            pd.Categorical(predicciones["class_name"], categories=clases),
            pd.Categorical(predicciones["pred_class_name"], categories=clases),
            dropna=False,
        )
        confusion = confusion.reindex(index=clases, columns=clases, fill_value=0)
        confusion.index.name = "Clase real"
        matriz_confusion(confusion.reset_index(), "CNN propia: matriz derivada de predicciones")
        st.caption(f"Predicciones registradas: {len(predicciones):,}.")
    else:
        st.info("No se encontró el CSV de predicciones para obtener la matriz de confusión CNN.")

with tab_lstm:
    st.subheader("Predicción de casos de dengue")
    comparacion = cargar(DATA / "metricas_lstm.csv")
    if comparacion is not None and not comparacion.empty:
        tabla(comparacion)
        if {"Modelo", "RMSE", "MAE"}.issubset(comparacion.columns):
            largo = comparacion.melt(id_vars="Modelo", value_vars=["RMSE", "MAE"],
                                     var_name="Métrica", value_name="Error")
            largo["Error"] = pd.to_numeric(largo["Error"], errors="coerce")
            grafico = alt.Chart(largo).mark_bar().encode(
                x=alt.X("Modelo:N", title=None, axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("Error:Q", title="Error (casos)"),
                color="Métrica:N", xOffset="Métrica:N",
                tooltip=["Modelo:N", "Métrica:N", alt.Tooltip("Error:Q", format=".2f")],
            ).properties(height=340, title="Comparación RMSE y MAE (menor es mejor)")
            st.altair_chart(grafico, width="stretch")
    else:
        st.info("No se encontró `metricas_lstm.csv`.")

    st.markdown("#### Validación cruzada y candidatos")
    resumen = cargar(DATA / "cv_resultados_resumen.csv")
    detalle = cargar(DATA / "cv_resultados_detalle.csv")
    if resumen is not None and not resumen.empty:
        tabla(resumen)
        if {"Modelo", "RMSE_mean"}.issubset(resumen.columns):
            mostrar = resumen[["Modelo", "RMSE_mean"]].copy()
            mostrar["RMSE_mean"] = pd.to_numeric(mostrar["RMSE_mean"], errors="coerce")
            grafico_barras(mostrar, "Modelo", "RMSE_mean", "RMSE promedio por candidato")
    else:
        st.info("No se encontró `cv_resultados_resumen.csv`.")
    if detalle is not None and not detalle.empty:
        with st.expander("Ver resultados por fold y semilla"):
            tabla(detalle)

with tab_ann:
    st.subheader("Clasificación de recomendaciones preventivas")
    metricas_ann = cargar(DATA / "metricas_ann_v2.csv")
    if metricas_ann is not None and not metricas_ann.empty:
        if "accuracy_global" in metricas_ann.columns:
            accuracy = numero(metricas_ann["accuracy_global"].iloc[0])
            if accuracy is not None:
                st.metric("Accuracy de prueba", f"{accuracy:.2%}")
        tabla(metricas_ann)
        if {"clase", "f1_score"}.issubset(metricas_ann.columns):
            solo_clases = metricas_ann[~metricas_ann["clase"].isin(["macro avg", "weighted avg"])].copy()
            solo_clases["f1_score"] = pd.to_numeric(solo_clases["f1_score"], errors="coerce")
            grafico_barras(solo_clases, "clase", "f1_score", "F1-score por recomendación")
    else:
        st.info("No se encontró `metricas_ann_v2.csv`.")

    st.markdown("#### Matriz de confusión")
    matriz = cargar(DATA / "matriz_confusion_ann_v2.csv")
    if matriz is not None:
        matriz_confusion(matriz, "ANN: matriz de confusión de prueba")
    else:
        st.info("No se encontró la matriz de confusión ANN.")

    st.markdown("#### Evolución del entrenamiento")
    historial = cargar(DATA / "historial_entrenamiento_ann_v2.csv")
    if historial is not None:
        grafico_historial(historial)
    else:
        st.info("No se encontró el historial de entrenamiento ANN.")

st.divider()
st.caption(
    "Los resultados se calculan sobre los conjuntos indicados en cada archivo. "
    "No compare accuracy de clasificación con RMSE/MAE de pronóstico: miden tareas distintas."
)
