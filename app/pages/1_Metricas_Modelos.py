"""Dashboard de métricas DEMO de VectorIA. Sustituir DATA_DEMO al recibir resultados reales."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.theme import aplicar_tema, mostrar_identidad, mostrar_encabezado

st.set_page_config(page_title="Métricas de modelos | VectorIA", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
aplicar_tema()
mostrar_identidad()
mostrar_encabezado("Métricas de modelos", "Evaluación de CNN, LSTM y ANN para la vigilancia del dengue.")
st.warning("**Resultados de demostración:** todas las métricas y gráficas de esta página son simuladas. No representan modelos entrenados ni resultados de evaluación reales.", icon="🧪")

AZUL = "#1E376C"
ROJO = "#EF4135"
AZUL_CLARO = "#4C83C5"
GRIS = "#52647D"

# DATOS TEMPORALES. Al integrar los modelos, reemplazar estos diccionarios
# por las métricas evaluadas en un conjunto de prueba independiente.
CNN = {
    "CNN propia": {"Accuracy": 87.2, "Precision": 85.4, "Recall": 84.1, "F1-score": 84.7},
    "MobileNetV2": {"Accuracy": 91.2, "Precision": 89.5, "Recall": 90.8, "F1-score": 90.1},
}
LSTM = {
    "SimpleRNN": {"RMSE": 180.2, "MAE": 135.7},
    "LSTM": {"RMSE": 125.4, "MAE": 91.8},
}
ANN = {"Accuracy": 88.0, "Precision": 88.5, "Recall": 86.2, "F1-score": 87.3}
SEMANAS = [f"S{i}" for i in range(1, 13)]
REAL = [620, 710, 660, 790, 840, 750, 900, 880, 970, 860, 790, 940]
PRED_LSTM = [590, 680, 690, 755, 820, 770, 880, 905, 940, 875, 815, 915]
PRED_RNN = [540, 610, 740, 695, 740, 830, 790, 960, 840, 930, 715, 850]

SECCIONES = {
    "cnn": ["Resumen", "Comparación de métricas", "Matrices de confusión", "Tabla de resultados"],
    "lstm": ["Resumen", "Comparación de errores", "Casos reales y predicciones", "Tabla de resultados"],
    "ann": ["Resumen", "Gráfico de rendimiento", "Matriz de confusión", "Tabla de resultados"],
}


def inicializar_estado():
    for modulo, secciones in SECCIONES.items():
        for indice in range(len(secciones)):
            clave = f"met_{modulo}_{indice}"
            if clave not in st.session_state:
                st.session_state[clave] = True


def cambiar_todas(modulo, estado):
    for indice in range(len(SECCIONES[modulo])):
        st.session_state[f"met_{modulo}_{indice}"] = estado


def cabecera_controles(modulo):
    col1, col2, resto = st.columns([1, 1, 3], vertical_alignment="center")
    with col1:
        st.button("⤢ Expandir todas", key=f"abrir_{modulo}", on_click=cambiar_todas, args=(modulo, True), use_container_width=True)
    with col2:
        st.button("⤡ Contraer todas", key=f"cerrar_{modulo}", on_click=cambiar_todas, args=(modulo, False), use_container_width=True)
    with resto:
        st.caption("Cada sección también se puede abrir o cerrar individualmente.")


def seccion(modulo, indice, descripcion=None):
    """Crea un contenedor grande con borde negro y botón de expansión."""
    clave = f"met_{modulo}_{indice}"
    titulo = SECCIONES[modulo][indice]
    contenedor = st.container(border=True)
    with contenedor:
        columna_titulo, columna_boton = st.columns([6, 1.5], vertical_alignment="center")
        with columna_titulo:
            st.markdown(f"#### {titulo}")
            if descripcion:
                st.caption(descripcion)
        with columna_boton:
            abierto = st.session_state[clave]
            if st.button("▴ Ocultar" if abierto else "▾ Mostrar", key=f"toggle_{modulo}_{indice}", use_container_width=True):
                st.session_state[clave] = not abierto
                st.rerun()
    return contenedor, st.session_state[clave]


def tarjetas(valores):
    columnas = st.columns(len(valores), gap="medium")
    for columna, (etiqueta, valor) in zip(columnas, valores.items()):
        with columna:
            st.metric(etiqueta, valor)


def estilo(fig, altura=410):
    fig.update_layout(
        height=altura, template="plotly_white",
        font=dict(color="#17243B"),
        margin=dict(l=12, r=15, t=28, b=18),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=1.14, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E4EAF2", zerolinecolor="#A8B5C7")
    return fig


def barras_comparadas(datos, nombres, unidades="%", minimo=0, maximo=None):
    fig = go.Figure()
    colores = [AZUL, ROJO, AZUL_CLARO]
    for indice, (modelo, metricas) in enumerate(datos.items()):
        fig.add_trace(go.Bar(name=modelo, x=nombres, y=[metricas[n] for n in nombres],
                             marker_color=colores[indice % len(colores)],
                             text=[f"{metricas[n]:.1f}{unidades}" for n in nombres], textposition="outside"))
    fig.update_layout(barmode="group", yaxis_title="Porcentaje (%)" if unidades == "%" else "Error (casos)")
    fig.update_yaxes(range=[minimo, maximo] if maximo is not None else None)
    return estilo(fig)


def matriz_confusion(matriz, clases, titulo):
    fig = go.Figure(go.Heatmap(
        z=matriz, x=clases, y=clases, colorscale="Blues", showscale=True,
        hovertemplate="Real: %{y}<br>Predicción: %{x}<br>Ejemplos: %{z}<extra></extra>",
        colorbar=dict(title="Conteo", thickness=14),
    ))
    for fila in range(len(clases)):
        for columna in range(len(clases)):
            valor = int(matriz[fila][columna])
            fig.add_annotation(x=clases[columna], y=clases[fila], text=str(valor),
                               showarrow=False, font=dict(color="white" if valor >= np.max(matriz)*.52 else "#17243B", size=15))
    fig.update_layout(title=dict(text=titulo, font=dict(size=15)),
                      xaxis_title="Predicción", yaxis_title="Clase real")
    fig.update_yaxes(autorange="reversed")
    return estilo(fig, altura=390)


def mostrar_grafico(fig, key):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": False}, key=key)


inicializar_estado()
tab_cnn, tab_lstm, tab_ann = st.tabs(["🦟 CNN · Mosquitos", "📈 LSTM · Dengue", "🛡️ ANN · Prevención"])

with tab_cnn:
    st.markdown("### Clasificación de especies de mosquitos")
    st.caption("CNN propia frente a MobileNetV2 · Métricas simuladas sobre una evaluación ilustrativa")
    cabecera_controles("cnn")

    panel, abierto = seccion("cnn", 0, "Indicadores destacados del modelo de demostración MobileNetV2")
    if abierto:
        with panel:
            tarjetas({nombre: f"{valor:.1f}%" for nombre, valor in CNN["MobileNetV2"].items()})
            st.caption("Estos indicadores son ficticios; se reemplazarán por resultados reales.")

    panel, abierto = seccion("cnn", 1, "Comparación de Accuracy, Precision, Recall y F1-score")
    if abierto:
        with panel:
            mostrar_grafico(barras_comparadas(CNN, list(CNN["CNN propia"].keys()), maximo=105), "graf_cnn")
            st.caption("Los porcentajes más altos suelen indicar mejor rendimiento; se debe revisar también el equilibrio entre clases.")

    panel, abierto = seccion("cnn", 2, "Ejemplos de matriz de confusión; filas = clase real, columnas = predicción")
    if abierto:
        with panel:
            modelo_cnn = st.selectbox("Modelo", list(CNN.keys()), key="modelo_mc_cnn")
            matrices = {
                "CNN propia": [[95, 12], [18, 85]],
                "MobileNetV2": [[101, 6], [11, 92]],
            }
            mostrar_grafico(matriz_confusion(matrices[modelo_cnn], ["Aedes", "Culex"], f"Matriz de confusión · {modelo_cnn}"), "mc_cnn")
            st.caption("Matriz ilustrativa binaria. Se adaptará a las especies efectivas del modelo CNN y sus etiquetas reales.")

    panel, abierto = seccion("cnn", 3, "Comparación tabular para consulta y descarga")
    if abierto:
        with panel:
            tabla = pd.DataFrame(CNN).T.rename_axis("Modelo").reset_index()
            st.dataframe(tabla, hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.NumberColumn(k, format="%.1f%%") for k in CNN["CNN propia"]})
            st.download_button("Descargar CSV demostrativo", tabla.to_csv(index=False).encode("utf-8-sig"),
                               "vectoria_cnn_demo.csv", "text/csv", key="dl_cnn")

with tab_lstm:
    st.markdown("### Predicción semanal de casos de dengue")
    st.caption("SimpleRNN frente a LSTM · Valores simulados, no son predicciones reales de 2025 o 2026")
    cabecera_controles("lstm")

    panel, abierto = seccion("lstm", 0, "Errores de ejemplo del modelo LSTM")
    if abierto:
        with panel:
            tarjetas({"RMSE · LSTM": f"{LSTM['LSTM']['RMSE']:.1f} casos", "MAE · LSTM": f"{LSTM['LSTM']['MAE']:.1f} casos"})
            st.caption("En RMSE y MAE, los valores menores indican menos error. No son porcentajes.")

    panel, abierto = seccion("lstm", 1, "Comparación de RMSE y MAE entre modelos")
    if abierto:
        with panel:
            mostrar_grafico(barras_comparadas(LSTM, ["RMSE", "MAE"], unidades=""), "graf_lstm_errores")
            st.caption("Los dos modelos deben evaluarse sobre el mismo conjunto de prueba para comparar sus errores.")

    panel, abierto = seccion("lstm", 2, "Serie temporal de ejemplo: valores observados y estimaciones")
    if abierto:
        with panel:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=SEMANAS, y=REAL, mode="lines+markers", name="Casos reales (demo)", line=dict(color=AZUL, width=3)))
            fig.add_trace(go.Scatter(x=SEMANAS, y=PRED_LSTM, mode="lines+markers", name="LSTM (demo)", line=dict(color=ROJO, width=2)))
            fig.add_trace(go.Scatter(x=SEMANAS, y=PRED_RNN, mode="lines+markers", name="SimpleRNN (demo)", line=dict(color=AZUL_CLARO, width=2, dash="dash")))
            fig.update_layout(xaxis_title="Semanas ilustrativas", yaxis_title="Casos de dengue")
            mostrar_grafico(estilo(fig, altura=440), "graf_lstm_lineas")
            st.caption("Las semanas S1–S12 son ficticias. Esta gráfica no calcula RMSE/MAE ni representa un pronóstico operativo.")

    panel, abierto = seccion("lstm", 3, "Tabla de métricas y detalle semanal de ejemplo")
    if abierto:
        with panel:
            st.markdown("**Métricas de error**")
            tabla = pd.DataFrame(LSTM).T.rename_axis("Modelo").reset_index()
            st.dataframe(tabla, hide_index=True, use_container_width=True)
            st.markdown("**Comparación por semana (demostración)**")
            serie = pd.DataFrame({"Semana": SEMANAS, "Real": REAL, "LSTM": PRED_LSTM, "SimpleRNN": PRED_RNN})
            st.dataframe(serie, hide_index=True, use_container_width=True)
            st.download_button("Descargar serie CSV demostrativa", serie.to_csv(index=False).encode("utf-8-sig"),
                               "vectoria_lstm_demo.csv", "text/csv", key="dl_lstm")

with tab_ann:
    st.markdown("### Recomendaciones preventivas")
    st.caption("Evaluación ilustrativa de un modelo ANN de clasificación")
    cabecera_controles("ann")

    panel, abierto = seccion("ann", 0, "Indicadores principales de clasificación")
    if abierto:
        with panel:
            tarjetas({nombre: f"{valor:.1f}%" for nombre, valor in ANN.items()})
            st.caption("Valores de demostración, no corresponden a la evaluación de la ANN del equipo.")

    panel, abierto = seccion("ann", 1, "Accuracy, Precision, Recall y F1-score")
    if abierto:
        with panel:
            fig = go.Figure(go.Bar(x=list(ANN.keys()), y=list(ANN.values()),
                                   marker_color=[AZUL, AZUL_CLARO, ROJO, "#CA6E27"],
                                   text=[f"{x:.1f}%" for x in ANN.values()], textposition="outside"))
            fig.update_layout(yaxis_title="Porcentaje (%)", showlegend=False)
            fig.update_yaxes(range=[0, 105])
            mostrar_grafico(estilo(fig), "graf_ann")

    panel, abierto = seccion("ann", 2, "Ejemplo binario pendiente de ajustar a las categorías de acción reales")
    if abierto:
        with panel:
            mostrar_grafico(matriz_confusion([[86, 14], [12, 88]], ["Acción A", "Acción B"], "Matriz de confusión · ANN (demo)"), "mc_ann")
            st.caption("Las clases A/B y sus conteos son ficticios. Si la ANN es multietiqueta o genera varias recomendaciones, esta visualización se adaptará.")

    panel, abierto = seccion("ann", 3, "Resultados resumidos y exportables")
    if abierto:
        with panel:
            tabla = pd.DataFrame([{"Métrica": m, "Resultado (%)": v} for m, v in ANN.items()])
            st.dataframe(tabla, hide_index=True, use_container_width=True,
                         column_config={"Resultado (%)": st.column_config.NumberColumn(format="%.1f%%")})
            st.download_button("Descargar CSV demostrativo", tabla.to_csv(index=False).encode("utf-8-sig"),
                               "vectoria_ann_demo.csv", "text/csv", key="dl_ann")

st.divider()
st.caption("VectorIA · Interfaz de métricas en fase de demostración. Sustituir métricas, matrices y series temporales después de evaluar los modelos reales.")
