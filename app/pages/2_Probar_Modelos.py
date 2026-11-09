"""Laboratorio de modelos de VectorIA: CNN, LSTM y ANN."""

import os
import sys
import math
import hashlib
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.theme import aplicar_tema, mostrar_identidad, mostrar_encabezado

API_URL = os.getenv("VECTORIA_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Probar modelos | VectorIA",
    page_icon="🦟",
    layout="wide",
)

aplicar_tema()
mostrar_identidad()

# Los campos de solo lectura permanecen opacos y legibles.
st.markdown(
    """
    <style>
    div[data-testid="stNumberInput"] input:disabled,
    div[data-testid="stTextInput"] input:disabled {
        opacity: 1 !important;
        -webkit-text-fill-color: #172b4d !important;
        color: #172b4d !important;
        background-color: #eaf0f6 !important;
        border: 1px solid #9aaec4 !important;
        font-weight: 600 !important;
        cursor: not-allowed !important;
    }
    div[data-testid="stNumberInput"] input:disabled::placeholder,
    div[data-testid="stTextInput"] input:disabled::placeholder {
        opacity: 1 !important;
        color: #526579 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. ESTADO DE LA APLICACIÓN
# ============================================================

for clave in (
    "cnn_resultado",
    "cnn_imagen_firma",
    "lstm_resultado",
    "ann_resultado",
    "ann_contexto_preparado",
    "ann_contexto_clave",
):
    if clave not in st.session_state:
        st.session_state[clave] = None

# ============================================================
# 3. COMUNICACIÓN CON FASTAPI
# ============================================================

def consultar_api(endpoint, datos=None, archivo=None):
    """Envía solicitudes a FastAPI y devuelve su respuesta."""

    try:
        if archivo is not None:
            respuesta = requests.post(
                f"{API_URL}{endpoint}",
                files={
                    "imagen": (
                        archivo.name,
                        archivo.getvalue(),
                        archivo.type,
                    )
                },
                timeout=60,
            )
        else:
            respuesta = requests.post(
                f"{API_URL}{endpoint}",
                json=datos,
                timeout=60,
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
    """Comprueba que la respuesta del modelo sea válida."""

    return (
        isinstance(resultado, dict)
        and resultado.get("estado") not in ("pendiente", "error")
    )


def extraer_especie(resultado):
    """Obtiene la especie identificada por la CNN."""

    if not isinstance(resultado, dict):
        return None

    return (
        resultado.get("especie")
        or resultado.get("clase")
        or resultado.get("prediccion")
    )


# ============================================================
# 4. FUNCIONES DEL LSTM
# ============================================================

def extraer_total_casos(resultado):
    """Suma los casos de las cuatro semanas pronosticadas."""

    if not modelo_disponible(resultado):
        return None

    predicciones = resultado.get("predicciones")

    if not isinstance(predicciones, list) or len(predicciones) != 4:
        return None

    valores = []

    for prediccion in predicciones:
        if isinstance(prediccion, dict):
            valor = prediccion.get(
                "casos_estimados",
                prediccion.get("casos"),
            )
        else:
            valor = prediccion

        if (
            not isinstance(valor, (int, float))
            or isinstance(valor, bool)
            or not math.isfinite(valor)
            or valor < 0
        ):
            return None

        valores.append(float(valor))

    total = sum(valores)

    if not math.isfinite(total):
        return None

    return total


def obtener_filas_prediccion(resultado):
    """Prepara las cuatro semanas para el gráfico y la tabla."""

    if not modelo_disponible(resultado):
        return []

    predicciones = resultado.get("predicciones", [])

    if not isinstance(predicciones, list) or len(predicciones) != 4:
        return []

    filas = []

    for indice, prediccion in enumerate(predicciones, start=1):

        if isinstance(prediccion, dict):
            semana = prediccion.get("semana", indice)
            casos = prediccion.get(
                "casos_estimados",
                prediccion.get("casos"),
            )
            fecha_texto = prediccion.get("fecha")
        else:
            semana = indice
            casos = prediccion
            fecha_texto = None

        if (
            not isinstance(casos, (int, float))
            or isinstance(casos, bool)
            or not math.isfinite(casos)
            or casos < 0
        ):
            return []

        fecha_formateada = None

        if fecha_texto:
            try:
                fecha_formateada = pd.to_datetime(
                    fecha_texto
                ).strftime("%d/%m/%Y")
            except (ValueError, TypeError):
                pass

        filas.append({
            "Semana": f"Semana {semana}",
            "Fecha": fecha_formateada or "No disponible",
            "Casos estimados": int(round(float(casos))),
        })

    return filas


def obtener_datos_para_ann(resultado_lstm):
    """Extrae únicamente los datos que la ANN necesita recibir.

    La ANN obtiene su propio umbral, lluvia, temperatura
    y semana epidemiológica desde el backend.
    """

    if not modelo_disponible(resultado_lstm):
        return None

    total = extraer_total_casos(resultado_lstm)

    if total is None:
        return None

    if not float(total).is_integer():
        return None

    predicciones = resultado_lstm.get("predicciones", [])

    if len(predicciones) != 4:
        return None

    primera_prediccion = predicciones[0]

    if not isinstance(primera_prediccion, dict):
        return None

    fecha_texto = primera_prediccion.get("fecha")

    if not isinstance(fecha_texto, str):
        return None

    try:
        fecha_referencia = date.fromisoformat(fecha_texto)
    except (ValueError, TypeError):
        return None

    return {
        "casos_pred_4sem": int(total),
        "fecha_referencia": fecha_referencia.isoformat(),
    }


def obtener_contexto_ann(resultado_ann):
    """Obtiene las variables históricas devueltas por la ANN."""

    if not modelo_disponible(resultado_ann):
        return None

    contexto = resultado_ann.get("contexto_ann")

    if not isinstance(contexto, dict):
        return None

    campos = (
        "casos_pred_4sem",
        "umbral_endemico",
        "lluvia_14d_mm",
        "temp_max",
        "semana_epi",
    )

    for campo in campos:
        valor = contexto.get(campo)

        if (
            isinstance(valor, bool)
            or not isinstance(valor, (int, float))
            or not math.isfinite(valor)
        ):
            return None

    return contexto


# ============================================================
# 5. ENCABEZADO
# ============================================================

mostrar_encabezado(
    "Laboratorio de modelos",
    "Pruebe los modelos de inteligencia artificial de VectorIA, "
    "consulte sus resultados y genere recomendaciones preventivas.",
)

st.divider()


# ============================================================
# 6. PESTAÑAS
# ============================================================

tab_cnn, tab_lstm, tab_ann = st.tabs([
    "🔬 Identificación CNN",
    "📈 Predicción LSTM",
    "🛡️ Recomendaciones ANN",
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
        gap="large",
    )

    with columna_entrada:

        st.markdown("#### Imagen de entrada")

        imagen = st.file_uploader(
            "Seleccione una imagen",
            type=["jpg", "jpeg", "png"],
            key="imagen_cnn",
        )

        # Si se cambia o elimina la fotografía, invalidar la clasificación anterior.
        firma_imagen = (
            hashlib.sha256(imagen.getvalue()).hexdigest()
            if imagen is not None else None
        )
        if st.session_state.cnn_imagen_firma != firma_imagen:
            st.session_state.cnn_imagen_firma = firma_imagen
            st.session_state.cnn_resultado = None
            st.session_state.ann_resultado = None

        if imagen is not None:
            st.image(
                imagen,
                caption="Imagen seleccionada",
                use_container_width=True,
            )

        if st.button(
            "Identificar mosquito",
            type="primary",
            use_container_width=True,
            disabled=imagen is None,
        ):
            st.session_state.cnn_resultado = None
            st.session_state.ann_resultado = None

            with st.spinner("Analizando imagen..."):
                resultado = consultar_api(
                    "/cnn/identificar-mosquito",
                    archivo=imagen,
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
                    "El modelo CNN no está disponible.",
                )
            )

        else:
            especie = extraer_especie(resultado)
            confianza = resultado.get("confianza")

            if especie is not None:
                st.markdown(
                    '<p class="small-label">Especie identificada</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {especie}")

            if (
                isinstance(confianza, (int, float))
                and math.isfinite(confianza)
            ):
                st.metric(
                    "Confianza",
                    f"{confianza * 100:.1f}%",
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
                        columns=["Especie", "Probabilidad"],
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
        gap="large",
    )

    with columna_control:

        st.markdown("#### Parámetros")

        fecha = st.date_input(
            "Semana de referencia",
            value=date(2024, 12, 30),
            format="DD/MM/YYYY",
            key="fecha_lstm",
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
            use_container_width=True,
        ):

            # Limpiar las predicciones anteriores.
            st.session_state.lstm_resultado = None
            st.session_state.ann_resultado = None
            st.session_state.ann_contexto_preparado = None
            st.session_state.ann_contexto_clave = None

            with st.spinner("Consultando modelo LSTM..."):
                resultado = consultar_api(
                    "/lstm/predecir-dengue",
                    datos={
                        "fecha": fecha.isoformat(),
                    },
                )

            if resultado is not None:
                st.session_state.lstm_resultado = resultado

                # La preparación pertenece a ANN y se solicita aparte.
                # El endpoint LSTM no agrega contexto_ann.
                datos_para_ann = obtener_datos_para_ann(resultado)

                if datos_para_ann is not None:
                    with st.spinner("Preparando variables de ANN..."):
                        preparacion = consultar_api(
                            "/ann/preparar-variables",
                            datos=datos_para_ann,
                        )

                    contexto = obtener_contexto_ann(preparacion)
                    if contexto is not None:
                        st.session_state.ann_contexto_preparado = contexto
                        st.session_state.ann_contexto_clave = (
                            datos_para_ann["casos_pred_4sem"],
                            datos_para_ann["fecha_referencia"],
                        )
                    else:
                        st.warning(
                            "El LSTM predijo los casos, pero no se pudieron "
                            "preparar las variables históricas de ANN."
                        )

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
                    "El modelo LSTM no está disponible.",
                )
            )

        else:
            filas = obtener_filas_prediccion(resultado)

            if filas:
                df = pd.DataFrame(filas)

                # Mantener el gráfico original.
                df_grafico = df.copy()

                df_grafico["Fecha corta"] = (
                    df_grafico["Fecha"].apply(
                        lambda valor: (
                            pd.to_datetime(
                                valor,
                                dayfirst=True,
                            ).strftime("%d/%m/%Y")
                            if valor != "No disponible"
                            else None
                        )
                    )
                )

                df_grafico["Semana visible"] = [
                    f"Sem. {i + 1} · "
                    f"{fecha_txt if fecha_txt else 'sin fecha'}"
                    for i, fecha_txt in enumerate(
                        df_grafico["Fecha corta"]
                    )
                ]

                orden_semanas = (
                    df_grafico["Semana visible"].tolist()
                )

                grafico_base = alt.Chart(df_grafico).encode(
                    x=alt.X(
                        "Semana visible:N",
                        sort=orden_semanas,
                        title="Semana y fecha del pronóstico",
                        axis=alt.Axis(
                            labelAngle=0,
                            labelLimit=180,
                        ),
                    ),
                    y=alt.Y(
                        "Casos estimados:Q",
                        title="Casos de dengue estimados",
                        scale=alt.Scale(zero=True),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Semana:N",
                            title="Semana",
                        ),
                        alt.Tooltip(
                            "Fecha:N",
                            title="Fecha",
                        ),
                        alt.Tooltip(
                            "Casos estimados:Q",
                            title="Casos",
                            format=",.0f",
                        ),
                    ],
                )

                linea = grafico_base.mark_line(
                    strokeWidth=3
                )

                puntos = grafico_base.mark_circle(
                    size=100
                )

                valores = grafico_base.mark_text(
                    dy=-16,
                    fontSize=13,
                    fontWeight="bold",
                ).encode(
                    text=alt.Text(
                        "Casos estimados:Q",
                        format=",.0f",
                    )
                )

                st.altair_chart(
                    (linea + puntos + valores).properties(
                        height=330
                    ),
                    use_container_width=True,
                )

                st.dataframe(
                    df[[
                        "Semana",
                        "Fecha",
                        "Casos estimados",
                    ]],
                    hide_index=True,
                    use_container_width=True,
                )

                total = extraer_total_casos(resultado)

                if total is not None:
                    st.metric(
                        "Total pronosticado (4 semanas)",
                        f"{total:,.0f} casos",
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
        "La ANN combina los casos pronosticados por LSTM "
        "con referencias históricas de clima y dengue, "
        "además de la detección de Aedes."
    )

    resultado_cnn = st.session_state.cnn_resultado
    resultado_lstm = st.session_state.lstm_resultado

    especie = None

    if modelo_disponible(resultado_cnn):
        especie = extraer_especie(resultado_cnn)

    # Los casos y la fecha vienen del LSTM; el contexto ya fue
    # solicitado a ANN automáticamente tras generar la predicción.
    datos_lstm_ann = obtener_datos_para_ann(resultado_lstm)
    contexto_ann = st.session_state.ann_contexto_preparado

    clave_actual = (
        datos_lstm_ann["casos_pred_4sem"],
        datos_lstm_ann["fecha_referencia"],
    ) if datos_lstm_ann is not None else None

    if (
        clave_actual is None
        or st.session_state.ann_contexto_clave != clave_actual
    ):
        contexto_ann = None

    contexto_listo = (
        datos_lstm_ann is not None
        and isinstance(contexto_ann, dict)
    )

    with st.container(border=True):
        st.markdown("#### Resultados de otros modelos")
        col_cnn, col_lstm = st.columns(2)

        with col_cnn:
            if especie is not None:
                st.success(f"CNN: {especie}")
            else:
                st.caption("CNN: sin resultado disponible")

        with col_lstm:
            if datos_lstm_ann is not None:
                st.success(
                    f"LSTM: {datos_lstm_ann['casos_pred_4sem']:,.0f} "
                    "casos previstos"
                )
            else:
                st.caption("LSTM: sin resultado disponible")

    # --------------------------------------------------------
    # Simulador ANN: conservar el contexto automático y editar una copia.
    # --------------------------------------------------------
    deteccion_cnn_disponible = especie in (
        "Aedes aegypti", "Aedes albopictus", "Culex"
    )
    aegypti_original = (
        "Sí" if especie == "Aedes aegypti" else "No"
    ) if deteccion_cnn_disponible else "Sin clasificar"

    # Reconfigurar el formulario solamente si cambia el pronóstico
    # o la clasificación CNN; no perder cambios al mover controles.
    origen_simulacion = (clave_actual, especie, contexto_listo)
    if st.session_state.get("ann_sim_origen") != origen_simulacion:
        st.session_state.ann_sim_origen = origen_simulacion
        if contexto_listo:
            st.session_state.ann_sim_casos = int(contexto_ann["casos_pred_4sem"])
            st.session_state.ann_sim_umbral = float(contexto_ann["umbral_endemico"])
            st.session_state.ann_sim_lluvia = float(contexto_ann["lluvia_14d_mm"])
            st.session_state.ann_sim_temperatura = float(contexto_ann["temp_max"])
        else:
            st.session_state.ann_sim_casos = 0
            st.session_state.ann_sim_umbral = 0.0
            st.session_state.ann_sim_lluvia = 0.0
            st.session_state.ann_sim_temperatura = 0.0
        st.session_state.ann_sim_aegypti = aegypti_original
        st.session_state.ann_resultado = None

    def invalidar_recomendacion_ann():
        """Evita mostrar como vigente una recomendación con datos antiguos."""
        st.session_state.ann_resultado = None

    def restablecer_simulador_ann():
        """Recupera las entradas originales sin modificar los modelos ni CSV."""
        if not contexto_listo:
            return
        st.session_state.ann_sim_casos = int(contexto_ann["casos_pred_4sem"])
        st.session_state.ann_sim_umbral = float(contexto_ann["umbral_endemico"])
        st.session_state.ann_sim_lluvia = float(contexto_ann["lluvia_14d_mm"])
        st.session_state.ann_sim_temperatura = float(contexto_ann["temp_max"])
        st.session_state.ann_sim_aegypti = aegypti_original
        st.session_state.ann_resultado = None

    with st.container(border=True):
        st.markdown("#### Variables de entrada · Simulador ¿Qué pasaría si?")
        st.caption(
            "Los valores se cargan automáticamente desde LSTM, CNN y las "
            "referencias históricas de ANN. Puede modificarlos para simular "
            "una recomendación. No se cambian las predicciones ni los datasets."
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            casos_simulados = st.number_input(
                "Casos pronosticados (4 semanas)",
                min_value=0,
                step=1,
                disabled=not contexto_listo,
                key="ann_sim_casos",
                on_change=invalidar_recomendacion_ann,
            )
            umbral_simulado = st.number_input(
                "Umbral endémico",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                disabled=not contexto_listo,
                key="ann_sim_umbral",
                on_change=invalidar_recomendacion_ann,
            )
            lluvia_simulada = st.number_input(
                "Lluvia acumulada en 14 días (mm)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                disabled=not contexto_listo,
                key="ann_sim_lluvia",
                on_change=invalidar_recomendacion_ann,
            )

        with col2:
            temperatura_simulada = st.number_input(
                "Temperatura máxima (°C)",
                min_value=-20.0,
                max_value=60.0,
                step=0.5,
                format="%.2f",
                disabled=not contexto_listo,
                key="ann_sim_temperatura",
                on_change=invalidar_recomendacion_ann,
            )
            st.text_input(
                "Fecha de referencia",
                value=(contexto_ann["fecha_referencia"]
                       if contexto_listo else "Sin predicción"),
                disabled=True,
            )
            st.number_input(
                "Semana epidemiológica",
                min_value=0,
                max_value=53,
                value=(int(contexto_ann["semana_epi"])
                       if contexto_listo else 0),
                step=1,
                disabled=True,
            )
            opcion_aegypti = st.selectbox(
                "¿Se detectó Aedes aegypti?",
                options=["Sin clasificar", "Sí", "No"],
                disabled=not contexto_listo,
                key="ann_sim_aegypti",
                on_change=invalidar_recomendacion_ann,
            )
            if deteccion_cnn_disponible:
                st.caption(
                    f"Detección automática de MobileNetV2: {especie}. "
                    "Puede cambiarla para simular otro escenario."
                )
            else:
                st.caption(
                    "Sin clasificación CNN: seleccione Sí o No solo si "
                    "desea simular manualmente la detección."
                )

        if datos_lstm_ann is None:
            st.info("Primero genere una predicción LSTM válida.")
        elif not contexto_listo:
            st.warning(
                "El LSTM devolvió casos y fecha, pero no se prepararon "
                "las variables de ANN. Genere nuevamente la predicción."
            )
        elif opcion_aegypti == "Sin clasificar":
            st.info(
                "Clasifique una imagen con MobileNetV2 o seleccione "
                "manualmente la detección para una simulación."
            )

        if contexto_listo:
            origen_clima = contexto_ann.get("origen_clima")
            if origen_clima == "registro_historico_ann":
                st.caption("Clima: registro histórico de vectoria_acciones.csv.")
            elif origen_clima == "referencia_climatica_historica_ann":
                st.caption(
                    "Clima: estimación histórica de vectoria_acciones.csv; "
                    "no es una observación futura."
                )
            advertencia = contexto_ann.get("advertencia")
            if advertencia:
                with st.expander("Información sobre el origen de los datos"):
                    st.write(advertencia)

        simulacion_modificada = contexto_listo and (
            int(casos_simulados) != int(contexto_ann["casos_pred_4sem"])
            or float(umbral_simulado) != float(contexto_ann["umbral_endemico"])
            or float(lluvia_simulada) != float(contexto_ann["lluvia_14d_mm"])
            or float(temperatura_simulada) != float(contexto_ann["temp_max"])
            or opcion_aegypti != aegypti_original
        )
        if simulacion_modificada:
            st.info(
                "Modo simulación: modificó valores originales. "
                "La siguiente recomendación usará los valores editados."
            )

        col_generar, col_restaurar = st.columns([2, 1], gap="medium")
        with col_generar:
            generar = st.button(
                "Generar recomendación",
                type="primary",
                use_container_width=True,
                disabled=not (contexto_listo and opcion_aegypti != "Sin clasificar"),
            )
        with col_restaurar:
            st.button(
                "Restablecer valores",
                use_container_width=True,
                disabled=not contexto_listo,
                on_click=restablecer_simulador_ann,
            )

        if generar:
            # El endpoint actual acepta las seis variables directamente.
            # La fecha y la semana epidemiológica siguen siendo coherentes.
            datos_ann = {
                "casos_pred_4sem": int(casos_simulados),
                "fecha_referencia": contexto_ann["fecha_referencia"],
                "aegypti_det": 1 if opcion_aegypti == "Sí" else 0,
                "umbral_endemico": float(umbral_simulado),
                "lluvia_14d_mm": float(lluvia_simulada),
                "temp_max": float(temperatura_simulada),
                "semana_epi": int(contexto_ann["semana_epi"]),
            }

            st.session_state.ann_resultado = None
            with st.spinner("Generando recomendación ANN..."):
                st.session_state.ann_resultado = consultar_api(
                    "/ann/recomendar-accion",
                    datos=datos_ann,
                )

    # --------------------------------------------------------
    # Resultado ANN
    # --------------------------------------------------------

    resultado = st.session_state.ann_resultado

    # --------------------------------------------------------
    # Resultado del modelo ANN
    # --------------------------------------------------------

    st.markdown("#### Resultado de la recomendación")

    if resultado is None:

        st.info(
            "Genere una predicción LSTM y clasifique una imagen con MobileNetV2 "
            "(o seleccione la detección manualmente para simular), "
            "y pulse Generar recomendación."
        )

    elif not modelo_disponible(resultado):

        st.warning(
            resultado.get(
                "mensaje",
                "El modelo ANN todavía no está disponible.",
            )
        )

    else:

        recomendacion = resultado.get("recomendacion")
        explicacion = resultado.get("explicacion")
        confianza = resultado.get("confianza")

        if recomendacion:

            st.success(
                "Acción recomendada: "
                + str(recomendacion)
                .replace("_", " ")
                .capitalize()
            )

            if isinstance(confianza, (int, float)):
                st.metric(
                    "Confianza del modelo",
                    f"{confianza * 100:.1f}%",
                )

            if explicacion:
                st.write(explicacion)

            probabilidades = resultado.get(
                "probabilidades"
            )

            if isinstance(probabilidades, dict):

                with st.expander(
                    "Ver probabilidades por recomendación"
                ):
                    tabla_probabilidades = pd.DataFrame(
                        [
                            {
                                "Recomendación": clase.replace(
                                    "_", " "
                                ).capitalize(),
                                "Probabilidad": float(valor),
                            }
                            for clase, valor in probabilidades.items()
                        ]
                    )

                    st.dataframe(
                        tabla_probabilidades,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Probabilidad": st.column_config.ProgressColumn(
                                "Probabilidad",
                                min_value=0,
                                max_value=1,
                                format="percent",
                            )
                        },
                    )

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

