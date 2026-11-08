
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Permite importar utils.theme tanto desde Home.py como desde esta página.
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.theme import aplicar_tema, mostrar_identidad, mostrar_encabezado


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

API_URL = os.getenv("VECTORIA_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Probar modelos | VectorIA",
    page_icon="🦟",
    layout="wide"
)


# ============================================================
# 2. IDENTIDAD VISUAL COMPARTIDA
# ============================================================

aplicar_tema()
mostrar_identidad()


# ============================================================
# 3. ESTADO DE LA APLICACIÓN
# ============================================================

if "cnn_resultado" not in st.session_state:
    st.session_state.cnn_resultado = None

if "lstm_resultado" not in st.session_state:
    st.session_state.lstm_resultado = None

if "ann_resultado" not in st.session_state:
    st.session_state.ann_resultado = None

valores_iniciales = {
    "ann_casos": 0.0,
    "ann_umbral": 0.0,
    "ann_lluvia": 0.0,
    "ann_temperatura": 0.0,
    "ann_aegypti": "No",
}

for clave, valor in valores_iniciales.items():
    if clave not in st.session_state:
        st.session_state[clave] = valor


# ============================================================
# 4. FUNCIONES DE COMUNICACIÓN CON LA API
# ============================================================

def consultar_api(endpoint, datos=None, archivo=None):
    """Envía solicitudes a FastAPI y devuelve el resultado."""

    try:
        if archivo is not None:
            respuesta = requests.post(
                f"{API_URL}{endpoint}",
                files={
                    "imagen": (
                        archivo.name,
                        archivo.getvalue(),
                        archivo.type
                    )
                },
                timeout=60
            )
        else:
            respuesta = requests.post(
                f"{API_URL}{endpoint}",
                json=datos,
                timeout=60
            )

        respuesta.raise_for_status()
        return respuesta.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con FastAPI. "
            "Compruebe que la API esté ejecutándose."
        )

    except requests.exceptions.Timeout:
        st.error("La API tardó demasiado en responder.")

    except requests.exceptions.HTTPError as error:
        detalle = error.response.text

        try:
            detalle = error.response.json().get("detail", detalle)
        except ValueError:
            pass

        st.error(f"Error de la API: {detalle}")

    except (ValueError, requests.exceptions.RequestException) as error:
        st.error(f"No se pudo procesar la respuesta: {error}")

    return None


def modelo_disponible(resultado):
    """Evita tratar las respuestas pendientes como predicciones."""

    return (
        isinstance(resultado, dict)
        and resultado.get("estado") not in ("pendiente", "error")
    )


def extraer_especie(resultado):
    """Obtiene la especie de una respuesta CNN."""

    return (
        resultado.get("especie")
        or resultado.get("clase")
        or resultado.get("prediccion")
    )


def extraer_total_casos(resultado):
    """Calcula el total pronosticado para las cuatro semanas."""

    total = resultado.get("casos_pred_4sem")

    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return float(total)

    predicciones = resultado.get("predicciones")

    if not isinstance(predicciones, list) or len(predicciones) != 4:
        return None

    valores = []

    for prediccion in predicciones:
        if isinstance(prediccion, dict):
            valor = prediccion.get("casos")
        else:
            valor = prediccion

        if not isinstance(valor, (int, float)):
            return None

        valores.append(float(valor))

    return sum(valores)


def obtener_filas_prediccion(resultado):
    """Prepara las predicciones LSTM para una tabla y gráfico."""

    predicciones = resultado.get("predicciones", [])

    if not isinstance(predicciones, list):
        return []

    filas = []

    for indice, prediccion in enumerate(predicciones, start=1):
        if isinstance(prediccion, dict):
            semana = prediccion.get("semana", f"Semana {indice}")
            casos = prediccion.get("casos")
        else:
            semana = f"Semana {indice}"
            casos = prediccion

        if isinstance(casos, (int, float)):
            filas.append({
                "Semana": str(semana),
                "Casos": float(casos)
            })

    return filas


# ============================================================
# 5. ENCABEZADO
# ============================================================

mostrar_encabezado(
    "Laboratorio de modelos",
    "Pruebe los modelos de inteligencia artificial de VectorIA, "
    "consulte sus resultados y genere recomendaciones preventivas."
)

st.divider()

# ============================================================
# 6. PESTAÑAS DE LOS MODELOS
# ============================================================

tab_cnn, tab_lstm, tab_ann = st.tabs([
    "🔬 Identificación CNN",
    "📈 Predicción LSTM",
    "🛡️ Recomendaciones ANN"
])


# ============================================================
# 7. MODELO CNN
# ============================================================

with tab_cnn:

    st.subheader("Identificación de mosquitos")
    st.caption(
        "Cargue una fotografía para identificar la especie "
        "mediante el modelo de clasificación CNN."
    )

    columna_entrada, columna_resultado = st.columns(
        [1, 1],
        gap="large"
    )

    with columna_entrada:

        st.markdown("#### Imagen de entrada")

        imagen = st.file_uploader(
            "Seleccione una imagen",
            type=["jpg", "jpeg", "png"],
            key="imagen_cnn"
        )

        if imagen is not None:
            st.image(
                imagen,
                caption="Imagen seleccionada",
                use_container_width=True
            )

        if st.button(
            "Identificar mosquito",
            type="primary",
            use_container_width=True,
            disabled=imagen is None
        ):

            st.session_state.cnn_resultado = None
            st.session_state.ann_resultado = None

            with st.spinner("Analizando imagen..."):
                resultado = consultar_api(
                    "/cnn/identificar-mosquito",
                    archivo=imagen
                )

            if resultado is not None:
                st.session_state.cnn_resultado = resultado

    with columna_resultado:

        st.markdown("#### Resultado del análisis")

        resultado = st.session_state.cnn_resultado

        if resultado is None:
            st.info(
                "Seleccione una imagen y ejecute el modelo "
                "para visualizar su clasificación."
            )

        elif not modelo_disponible(resultado):
            st.warning(
                resultado.get(
                    "mensaje",
                    "El modelo CNN no está disponible."
                )
            )

        else:
            especie = extraer_especie(resultado)
            confianza = resultado.get("confianza")

            if especie is not None:
                st.markdown(
                    '<p class="small-label">Especie identificada</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"### {especie}"
                )

            if isinstance(confianza, (int, float)):
                st.metric(
                    "Confianza",
                    f"{confianza * 100:.1f}%"
                )
                st.progress(
                    max(0.0, min(float(confianza), 1.0))
                )

            probabilidades = resultado.get("probabilidades")

            if isinstance(probabilidades, dict):
                st.markdown("#### Probabilidades por especie")
                st.bar_chart(
                    pd.DataFrame(
                        list(probabilidades.items()),
                        columns=["Especie", "Probabilidad"]
                    ).set_index("Especie")
                )

            if especie is None:
                st.info(
                    "La API respondió, pero todavía no devolvió "
                    "una especie identificada."
                )

    st.divider()

    st.caption(
        "La identificación es una clasificación automatizada "
        "y puede contener errores."
    )


# ============================================================
# 8. MODELO LSTM
# ============================================================

with tab_lstm:

    st.subheader("Predicción de casos de dengue")
    st.caption(
        "Consulte el pronóstico nacional de dengue "
        "para las próximas cuatro semanas."
    )

    columna_control, columna_grafico = st.columns(
        [1, 2],
        gap="large"
    )

    with columna_control:

        st.markdown("#### Parámetros")

        fecha = st.date_input(
            "Semana de referencia",
            value=date(2024, 12, 30),
            format="DD/MM/YYYY",
            key="fecha_lstm"
        )

        st.caption(
            "La disponibilidad de fechas dependerá "
            "de los datos utilizados para entrenar el modelo."
        )

        st.info(
            "El pronóstico corresponde a Costa Rica "
            "en general, no a una provincia o cantón."
        )

        if st.button(
            "Generar predicción",
            type="primary",
            use_container_width=True
        ):

            st.session_state.lstm_resultado = None
            st.session_state.ann_resultado = None

            with st.spinner("Consultando modelo LSTM..."):
                resultado = consultar_api(
                    "/lstm/predecir-dengue",
                    datos={
                        "fecha": fecha.isoformat()
                    }
                )

            if resultado is not None:
                st.session_state.lstm_resultado = resultado

    with columna_grafico:

        st.markdown("#### Pronóstico de cuatro semanas")

        resultado = st.session_state.lstm_resultado

        if resultado is None:
            st.info(
                "Seleccione una semana y ejecute el modelo "
                "para visualizar el pronóstico."
            )

        elif not modelo_disponible(resultado):
            st.warning(
                resultado.get(
                    "mensaje",
                    "El modelo LSTM no está disponible."
                )
            )

        else:
            filas = obtener_filas_prediccion(resultado)

            if filas:
                df = pd.DataFrame(filas)

                st.line_chart(
                    df.set_index("Semana")["Casos"],
                    use_container_width=True
                )

                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True
                )

                total = extraer_total_casos(resultado)

                if total is not None:
                    st.metric(
                        "Total pronosticado (4 semanas)",
                        f"{total:,.0f} casos"
                    )

            else:
                st.info(
                    "La API todavía no devolvió las cuatro "
                    "predicciones semanales en un formato compatible."
                )

    st.divider()

    st.caption(
        "Las predicciones son estimaciones estadísticas "
        "y no garantizan la cantidad real de casos futuros."
    )


# ============================================================
# 9. MODELO ANN
# ============================================================

with tab_ann:

    st.subheader("Recomendaciones preventivas")
    st.caption(
        "Combine los resultados de CNN y LSTM con las "
        "condiciones epidemiológicas y climáticas."
    )

    # --------------------------------------------------------
    # Importar resultados de CNN y LSTM
    # --------------------------------------------------------

    resultado_cnn = st.session_state.cnn_resultado
    resultado_lstm = st.session_state.lstm_resultado

    especie = None
    total_casos = None

    if resultado_cnn and modelo_disponible(resultado_cnn):
        especie = extraer_especie(resultado_cnn)

    if resultado_lstm and modelo_disponible(resultado_lstm):
        total_casos = extraer_total_casos(resultado_lstm)

    with st.container(border=True):

        st.markdown("#### Resultados de otros modelos")

        col_cnn, col_lstm = st.columns(2)

        with col_cnn:
            if especie is not None:
                st.success(f"CNN: {especie}")
            else:
                st.caption("CNN: sin resultado disponible")

        with col_lstm:
            if total_casos is not None:
                st.success(
                    f"LSTM: {total_casos:,.0f} casos previstos"
                )
            else:
                st.caption("LSTM: sin resultado disponible")

        if st.button(
            "Utilizar resultados disponibles",
            use_container_width=True
        ):

            importados = []

            if especie is not None:
                st.session_state.ann_aegypti = (
                    "Sí"
                    if "aegypti" in str(especie).lower()
                    else "No"
                )
                importados.append("CNN")

            if total_casos is not None:
                st.session_state.ann_casos = max(
                    0.0,
                    float(total_casos)
                )
                importados.append("LSTM")

            if importados:
                st.session_state.ann_resultado = None
                st.success(
                    "Datos importados desde: "
                    + ", ".join(importados)
                )
            else:
                st.warning(
                    "Todavía no hay resultados para importar."
                )

    # --------------------------------------------------------
    # Formulario ANN
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown("#### Variables de entrada")

        st.caption(
            "Puede modificar manualmente los valores "
            "para analizar diferentes escenarios."
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:

            casos = st.number_input(
                "Casos pronosticados (4 semanas)",
                min_value=0.0,
                step=1.0,
                key="ann_casos",
                help="Total de casos estimados por la LSTM."
            )

            umbral = st.number_input(
                "Umbral endémico",
                min_value=0.0,
                step=1.0,
                key="ann_umbral"
            )

            lluvia = st.number_input(
                "Lluvia acumulada en 14 días (mm)",
                min_value=0.0,
                step=1.0,
                key="ann_lluvia"
            )

        with col2:

            temperatura = st.number_input(
                "Temperatura máxima (°C)",
                min_value=-20.0,
                max_value=60.0,
                step=0.5,
                key="ann_temperatura"
            )

            aegypti = st.radio(
                "¿Se detectó Aedes aegypti?",
                options=["Sí", "No"],
                horizontal=True,
                key="ann_aegypti"
            )

            st.caption(
                "La detección puede introducirse manualmente "
                "o importarse desde CNN."
            )

        if st.button(
            "Generar recomendación",
            type="primary",
            use_container_width=True
        ):

            datos_ann = {
                "casos_pred_4sem": casos,
                "umbral_endemico": umbral,
                "lluvia_14d_mm": lluvia,
                "temp_max": temperatura,
                "aegypti_det": 1 if aegypti == "Sí" else 0
            }

            with st.spinner("Analizando escenario..."):
                resultado = consultar_api(
                    "/ann/recomendar-accion",
                    datos=datos_ann
                )

            st.session_state.ann_resultado = resultado

    # --------------------------------------------------------
    # Resultado ANN
    # --------------------------------------------------------

    st.markdown("#### Resultado de la recomendación")

    resultado = st.session_state.ann_resultado

    if resultado is None:
        st.info(
            "Complete las variables y ejecute el modelo ANN "
            "para obtener una recomendación."
        )

    elif not modelo_disponible(resultado):
        st.warning(
            resultado.get(
                "mensaje",
                "El modelo ANN todavía no está disponible."
            )
        )

    else:
        recomendacion = resultado.get("recomendacion")
        explicacion = resultado.get("explicacion")

        if recomendacion:
            st.success(
                "Acción recomendada: "
                + str(recomendacion).replace("_", " ").capitalize()
            )

            if explicacion:
                st.write(explicacion)
        else:
            st.info(
                "La API respondió, pero todavía no devolvió "
                "una recomendación."
            )

    st.divider()

    st.caption(
        "Las recomendaciones constituyen apoyo experimental "
        "para la vigilancia del dengue y no sustituyen "
        "las decisiones de las autoridades sanitarias."
    )


