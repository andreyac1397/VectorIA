"""Panel de métricas de VectorIA.

Ubicación: app/pages/1_Metricas_Modelos.py
Los datos se leen desde data/processed y no se inventan métricas ausentes.
"""
from pathlib import Path
import sys
import math

import matplotlib.pyplot as plt
import numpy as np
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

# Estilo institucional: fondo claro, contraste alto y bordes negros.
st.markdown("""
<style>
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] .stCaption {
    color: #263244 !important;
}
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 { color: #102956 !important; }
details[data-testid="stExpander"] {
    border: 1.5px solid #111827 !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,.75) !important;
}
details[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #12254b !important;
}
.vectoria-table-wrap {
    width: 100%; overflow-x: auto; margin: 16px auto 22px;
    border: 2px solid #111827; border-radius: 10px;
    background: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,.08);
}
.vectoria-table { width: 100%; border-collapse: collapse;
    font-size: 14px; color: #111827; text-align: center; }
.vectoria-table th {
    background: #1E376C; color: white !important;
    font-size: 13px; font-weight: 750; text-align: center;
    padding: 14px 11px; border: 1px solid #111827;
    white-space: normal; min-width: 105px;
}
.vectoria-table td {
    color: #172033 !important; padding: 11px 12px;
    border: 1px solid #374151; text-align: center; vertical-align: middle;
}
.vectoria-table tbody tr:nth-child(even) { background: #e9eff8; }
.vectoria-table tbody tr:hover { background: #dae6f6; }
.vectoria-table td:first-child {
    font-weight: 650; color: #102956 !important;
}


/* Sidebar: el tema compartido deja los textos oscuros sobre azul. */
[data-testid="stSidebar"],
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: #1E376C !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"],
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"],
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 650 !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] *,
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [aria-current="page"],
[data-testid="stSidebar"] [aria-selected="true"] {
    background-color: #345184 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [aria-current="page"] *,
[data-testid="stSidebar"] [aria-selected="true"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button * { color: #FFFFFF !important; }

</style>
""", unsafe_allow_html=True)


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
        n = float(valor)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def porcentaje(valor):
    n = numero(valor)
    return f"{n:.2%}" if n is not None else "—"


def decimal(valor, digitos=2):
    n = numero(valor)
    return f"{n:,.{digitos}f}" if n is not None else "—"


def tabla(df: pd.DataFrame):
    """Tabla HTML centrada, legible y sin dependencia de PyArrow."""
    if df is None or df.empty:
        st.info("No hay filas disponibles para esta tabla.")
        return
    vista = df.copy()
    # Solo presentación: nunca se alteran los datos de origen.
    for columna in vista.columns:
        if pd.api.types.is_numeric_dtype(vista[columna]):
            vista[columna] = vista[columna].map(
                lambda n: "—" if pd.isna(n) else (f"{n:,.4f}" if isinstance(n, float) else str(n))
            )
    tabla_html = vista.to_html(index=False, escape=True, border=0, classes="vectoria-table")
    st.markdown('<div class="vectoria-table-wrap">' + tabla_html + '</div>',
                unsafe_allow_html=True)


def crear_ejes(titulo: str, ancho: float = 9.5, alto: float = 4.7):
    fig, ax = plt.subplots(figsize=(ancho, alto), facecolor="#ffffff")
    fig.patch.set_edgecolor("#111111")
    fig.patch.set_linewidth(2.3)
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.3)
    ax.set_title(titulo, fontsize=13, fontweight="bold", color="#152a51", pad=18)
    ax.tick_params(colors="#111827", labelsize=10)
    ax.xaxis.label.set_color("#111827")
    ax.yaxis.label.set_color("#111827")
    return fig, ax


def mostrar_grafico(fig):
    fig.tight_layout(pad=2.1)
    st.pyplot(fig, clear_figure=False, use_container_width=True)
    plt.close(fig)


def grafico_barras(df: pd.DataFrame, eje_x: str, eje_y: str, titulo: str, ordenar=None):
    if df.empty:
        return
    datos = df.copy()
    datos[eje_y] = pd.to_numeric(datos[eje_y], errors="coerce")
    datos = datos.dropna(subset=[eje_y])
    if datos.empty:
        return
    if ordenar is not None:
        datos[eje_x] = pd.Categorical(datos[eje_x], categories=ordenar, ordered=True)
        datos = datos.sort_values(eje_x)
    fig, ax = crear_ejes(titulo, max(8, len(datos) * 1.45), 5)
    barras = ax.bar(datos[eje_x].astype(str), datos[eje_y], color="#1E376C",
                    edgecolor="#111111", linewidth=1.1, width=.56)
    ax.bar_label(barras, fmt="%.3f", padding=5, fontsize=9, color="#111827", fontweight="bold")
    ax.set_ylabel(eje_y, fontweight="bold")
    ax.grid(axis="y", alpha=.3, linestyle="--", color="#475569")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=20)
    ax.margins(y=.18)
    mostrar_grafico(fig)


def barras_agrupadas(df: pd.DataFrame, categoria: str, serie: str,
                     valor: str, titulo: str, etiqueta_y: str, porcentaje=False):
    if df.empty:
        return
    pivote = df.pivot_table(index=categoria, columns=serie, values=valor,
                            aggfunc="first", sort=False)
    if pivote.empty:
        return
    fig, ax = crear_ejes(titulo, max(9, len(pivote) * 2.05), 5.3)
    posiciones = np.arange(len(pivote))
    grupos = len(pivote.columns)
    ancho = min(.78 / max(grupos, 1), .34)
    colores = ["#1E376C", "#EF4135", "#117C83", "#7A4BA6", "#AD6800"]
    for indice, nombre in enumerate(pivote.columns):
        valores = pd.to_numeric(pivote[nombre], errors="coerce")
        barras = ax.bar(posiciones + (indice - (grupos - 1) / 2) * ancho,
                        valores, width=ancho, label=str(nombre),
                        color=colores[indice % len(colores)],
                        edgecolor="#111111", linewidth=1.0)
        etiquetas = [f"{v:.1%}" if porcentaje else f"{v:.1f}" if pd.notna(v) else ""
                     for v in valores]
        ax.bar_label(barras, labels=etiquetas, padding=4,
                     fontsize=8.5, color="#111827", fontweight="bold")
    ax.set_xticks(posiciones)
    ax.set_xticklabels([str(x) for x in pivote.index], rotation=12, ha="right")
    ax.set_ylabel(etiqueta_y, fontweight="bold")
    if porcentaje:
        ax.set_ylim(0, 1.13)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    else:
        ax.margins(y=.19)
    ax.grid(axis="y", linestyle="--", alpha=.28, color="#475569")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=min(4, grupos),
              frameon=True, facecolor="white", edgecolor="#111111", fontsize=9)
    fig.subplots_adjust(bottom=.22)
    mostrar_grafico(fig)


def matriz_confusion(df: pd.DataFrame, titulo: str):
    if df is None or df.empty:
        return
    matriz = df.copy()
    matriz = matriz.rename(columns={matriz.columns[0]: "Clase real"})
    matriz["Clase real"] = matriz["Clase real"].astype(str)
    columnas = [c for c in matriz.columns if c != "Clase real"]
    if not columnas:
        tabla(matriz)
        return
    valores = matriz[columnas].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    fig, ax = crear_ejes(titulo, max(7, len(columnas) * 1.7), max(4.6, len(matriz) * 1.1))
    imagen = ax.imshow(valores, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(columnas)))
    ax.set_xticklabels(columnas, rotation=30, ha="right")
    ax.set_yticks(range(len(matriz)))
    ax.set_yticklabels(matriz["Clase real"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_xlabel("Predicción", fontweight="bold")
    ax.set_ylabel("Valor real", fontweight="bold")
    umbral = valores.max() / 2 if valores.size else 0
    for i in range(len(matriz)):
        for j in range(len(columnas)):
            ax.text(j, i, f"{valores[i, j]:.0f}", ha="center", va="center",
                    color="white" if valores[i, j] > umbral else "black")
    ax.set_xticks(np.arange(-.5, len(columnas), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(matriz), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(imagen, ax=ax, label="Cantidad")
    mostrar_grafico(fig)


def grafico_historial(df: pd.DataFrame):
    requeridas = {"epoca", "accuracy", "val_accuracy", "loss", "val_loss"}
    if not requeridas.issubset(df.columns):
        st.info("El historial no contiene las columnas esperadas para graficar.")
        tabla(df)
        return
    epocas = pd.to_numeric(df["epoca"], errors="coerce")
    for variables, titulo in [(["accuracy", "val_accuracy"], "Accuracy por época"),
                              (["loss", "val_loss"], "Pérdida por época")]:
        fig, ax = crear_ejes(titulo, 9.5, 4.8)
        for i, variable in enumerate(variables):
            ax.plot(epocas, pd.to_numeric(df[variable], errors="coerce"),
                    label=variable, linewidth=2.4, marker="o", markersize=2.3,
                    color=["#1E376C", "#EF4135"][i])
        ax.set_xlabel("Época")
        ax.set_ylabel("Accuracy" if "Accuracy" in titulo else "Pérdida")
        ax.grid(alpha=.30, linestyle="--", color="#475569")
        ax.legend(frameon=True, edgecolor="#111111")
        mostrar_grafico(fig)


# Controles independientes dentro de cada pestaña.
# El cambio de generación reinicia la identidad de los desplegables de esa pestaña,
# conservando la apertura y cierre individual entre pulsaciones globales.
for _pestana in ("cnn", "lstm", "ann"):
    st.session_state.setdefault(f"metricas_estado_{_pestana}", None)
    st.session_state.setdefault(f"metricas_version_{_pestana}", 0)


def cambiar_expansion(pestana: str, abierto: bool):
    st.session_state[f"metricas_estado_{pestana}"] = abierto
    st.session_state[f"metricas_version_{pestana}"] += 1


def controles_pestana(pestana: str):
    izquierda, derecha, _ = st.columns([1.4, 1.4, 4.2])
    with izquierda:
        st.button("▾ Expandir todo", key=f"expandir_{pestana}",
                  on_click=cambiar_expansion, args=(pestana, True),
                  use_container_width=True)
    with derecha:
        st.button("▴ Colapsar todo", key=f"colapsar_{pestana}",
                  on_click=cambiar_expansion, args=(pestana, False),
                  use_container_width=True)


def desplegable(pestana: str, titulo: str, abierto_por_defecto: bool = False):
    actual = st.session_state[f"metricas_estado_{pestana}"]
    abierto = abierto_por_defecto if actual is None else actual
    version = st.session_state[f"metricas_version_{pestana}"]
    # Espacios invisibles fuerzan a Streamlit a aplicar el nuevo estado expanded.
    etiqueta = titulo + ("\u200b" * version)
    return st.expander(etiqueta, expanded=abierto)


tab_cnn, tab_lstm, tab_ann = st.tabs([
    "🔬 CNN y MobileNetV2", "📈 LSTM y comparativas", "🛡️ ANN"
])

with tab_cnn:
    controles_pestana("cnn")
    st.subheader("Clasificación de especies de mosquitos")
    st.caption("Las tablas y gráficas pueden abrirse individualmente o con los controles de esta pestaña.")
    final_path = CNN / "cnn_final_test_metrics.csv"
    final_cnn = cargar(final_path)
    columnas_finales = [
        "Modelo", "Accuracy_test", "Precision_macro_test",
        "Recall_macro_test", "F1_macro_test", "F1_Aedes_aegypti",
        "F1_Aedes_albopictus", "F1_Culex",
    ]
    if final_cnn is not None and not final_cnn.empty:
        faltantes = set(columnas_finales) - set(final_cnn.columns)
        if faltantes:
            st.warning(f"Faltan columnas en {final_path.name}: {', '.join(sorted(faltantes))}")
            with desplegable("cnn", "Tabla CSV de prueba final"):
                tabla(final_cnn)
        else:
            final_cnn = final_cnn[columnas_finales].copy()
            for nombre in columnas_finales[1:]:
                final_cnn[nombre] = pd.to_numeric(final_cnn[nombre], errors="coerce")
            final_cnn = final_cnn.dropna(subset=["Modelo", "Accuracy_test", "F1_macro_test"])
            st.markdown("#### Evaluación final: CNN propia vs. MobileNetV2")

            if not final_cnn.empty:
                visual = final_cnn.rename(columns={
                    "Accuracy_test": "Accuracy",
                    "Precision_macro_test": "Precision macro",
                    "Recall_macro_test": "Recall macro",
                    "F1_macro_test": "F1 macro",
                    "F1_Aedes_aegypti": "F1 Aedes aegypti",
                    "F1_Aedes_albopictus": "F1 Aedes albopictus",
                    "F1_Culex": "F1 Culex",
                })
                columnas_presentacion = [c for c in visual.columns if c != "Modelo"]
                with desplegable("cnn", "Tabla de métricas finales", True):
                    tabla(visual.assign(**{
                        c: visual[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
                        for c in columnas_presentacion
                    }))
                with desplegable("cnn", "Gráfico: comparación general", True):
                    clasificacion = visual.melt(
                        id_vars="Modelo",
                        value_vars=["Accuracy", "Precision macro", "Recall macro", "F1 macro"],
                        var_name="Métrica", value_name="Valor",
                    )
                    barras_agrupadas(clasificacion, "Métrica", "Modelo", "Valor",
                                     "Métricas de prueba final",
                                     "Puntuación", porcentaje=True)
                with desplegable("cnn", "Gráfico: F1-score por especie"):
                    especies = visual.melt(
                        id_vars="Modelo",
                        value_vars=["F1 Aedes aegypti", "F1 Aedes albopictus", "F1 Culex"],
                        var_name="Especie", value_name="F1",
                    )
                    barras_agrupadas(especies, "Especie", "Modelo", "F1",
                                     "F1-score por especie (prueba final)", "F1-score", porcentaje=True)
                if {"CNN propia", "MobileNetV2"}.issubset(set(visual["Modelo"])):
                    ganador = visual.loc[visual["Accuracy"].idxmax()]
                    st.caption(f"Mayor accuracy de prueba: {ganador['Modelo']} ({ganador['Accuracy']:.2%}).")
    else:
        st.info("Falta `data/processed/cnn/cnn_final_test_metrics.csv`.")

    st.markdown("#### Validación de la CNN propia")
    cnn_metrics = cargar(CNN / "cnn_validation_metrics.csv")
    with desplegable("cnn", "Tabla de métricas de validación"):
        if cnn_metrics is not None and not cnn_metrics.empty:
            st.caption("Resultados de validación, independientes de la prueba final.")
            tabla(cnn_metrics)
        else:
            st.info("No se encontró `cnn_validation_metrics.csv`.")

    with desplegable("cnn", "Matriz de confusión (validación)"):
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
            matriz_confusion(confusion.reset_index(), "CNN propia: matriz de validación")
            st.caption(f"Predicciones de validación registradas: {len(predicciones):,}.")
        else:
            st.info("No se encontró el CSV de predicciones de validación de CNN.")
        st.caption("No se presenta una matriz de prueba de MobileNetV2 sin sus predicciones individuales.")

with tab_lstm:
    controles_pestana("lstm")
    st.subheader("Predicción de casos de dengue")
    st.caption("RMSE y MAE: Skill positivo indica mejora frente a persistencia.")
    comparacion = cargar(DATA / "metricas_lstm.csv")
    with desplegable("lstm", "Tabla de métricas finales LSTM y comparativas", True):
        if comparacion is not None and not comparacion.empty:
            tabla(comparacion)
        else:
            st.info("No se encontró `metricas_lstm.csv`.")
    with desplegable("lstm", "Gráfico RMSE y MAE", True):
        if comparacion is not None and {"Modelo", "RMSE", "MAE"}.issubset(comparacion.columns):
            largo = comparacion.melt(id_vars="Modelo", value_vars=["RMSE", "MAE"],
                                     var_name="Métrica", value_name="Error")
            largo["Error"] = pd.to_numeric(largo["Error"], errors="coerce")
            barras_agrupadas(largo, "Modelo", "Métrica", "Error",
                             "Comparación RMSE y MAE (menor es mejor)", "Error (casos)")
        else:
            st.info("Faltan métricas para el gráfico.")

    st.markdown("#### Validación cruzada y candidatos")
    st.caption("Los folds y semillas no se mezclan con la evaluación final.")
    resumen = cargar(DATA / "cv_resultados_resumen.csv")
    detalle = cargar(DATA / "cv_resultados_detalle.csv")
    with desplegable("lstm", "Tabla resumen de validación cruzada"):
        if resumen is not None and not resumen.empty:
            tabla(resumen)
        else:
            st.info("No se encontró `cv_resultados_resumen.csv`.")
    with desplegable("lstm", "Gráfico RMSE promedio por candidato"):
        if resumen is not None and {"Modelo", "RMSE_mean"}.issubset(resumen.columns):
            mostrar = resumen[["Modelo", "RMSE_mean"]].copy()
            mostrar["RMSE_mean"] = pd.to_numeric(mostrar["RMSE_mean"], errors="coerce")
            grafico_barras(mostrar, "Modelo", "RMSE_mean", "RMSE promedio por candidato")
        else:
            st.info("Faltan métricas para el gráfico.")
    with desplegable("lstm", "Tabla de resultados por fold y semilla"):
        if detalle is not None and not detalle.empty:
            tabla(detalle)
        else:
            st.info("No se encontró `cv_resultados_detalle.csv`.")

with tab_ann:
    controles_pestana("ann")
    st.subheader("Clasificación de recomendaciones preventivas")
    metricas_ann = cargar(DATA / "metricas_ann_v2.csv")
    if metricas_ann is not None and not metricas_ann.empty and "accuracy_global" in metricas_ann.columns:
        accuracy = numero(metricas_ann["accuracy_global"].iloc[0])
        if accuracy is not None:
            st.metric("Accuracy de prueba", f"{accuracy:.2%}")
    with desplegable("ann", "Tabla de métricas ANN por clase", True):
        if metricas_ann is not None and not metricas_ann.empty:
            tabla(metricas_ann)
        else:
            st.info("No se encontró `metricas_ann_v2.csv`.")
    with desplegable("ann", "Gráfico F1-score por recomendación", True):
        if metricas_ann is not None and {"clase", "f1_score"}.issubset(metricas_ann.columns):
            solo_clases = metricas_ann[~metricas_ann["clase"].isin(["macro avg", "weighted avg"])].copy()
            solo_clases["f1_score"] = pd.to_numeric(solo_clases["f1_score"], errors="coerce")
            grafico_barras(solo_clases, "clase", "f1_score", "F1-score por recomendación")
        else:
            st.info("Faltan métricas ANN para graficar.")
    with desplegable("ann", "Matriz de confusión ANN"):
        matriz = cargar(DATA / "matriz_confusion_ann_v2.csv")
        if matriz is not None:
            matriz_confusion(matriz, "ANN: matriz de confusión de prueba")
        else:
            st.info("No se encontró la matriz de confusión ANN.")
        st.caption("La matriz debe corresponder a la versión evaluada de `ann_model_v2.keras`.")
    with desplegable("ann", "Curvas de entrenamiento ANN"):
        historial = cargar(DATA / "historial_entrenamiento_ann_v2.csv")
        if historial is not None:
            grafico_historial(historial)
        else:
            st.info("No se encontró el historial de entrenamiento ANN.")

st.divider()
st.caption(
    "Todos los gráficos se calculan a partir de los CSV existentes. "
    "No compare accuracy de clasificación con RMSE/MAE de pronóstico: miden tareas distintas."
)
