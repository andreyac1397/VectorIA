from pathlib import Path
import sys
import os

import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from utils.theme import aplicar_tema, mostrar_identidad, mostrar_encabezado

API_URL = os.getenv("VECTORIA_API_URL", "http://127.0.0.1:8000")
GITHUB_LOGO = APP_DIR / "assets" / "github.png"

st.set_page_config(page_title="Inicio | VectorIA", page_icon="🦟", layout="wide", initial_sidebar_state="expanded")
aplicar_tema()
mostrar_identidad()

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 1.8rem; }
    [data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: .75rem; }
    [data-testid="stMain"] h3 { margin-top: .3rem; margin-bottom: .35rem; }
    [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #000 !important;
        border-radius: 10px;
        background-color: #FFFFFF;
    }
    [data-testid="stPageLink"] a {
        background-color: #1E376C !important;
        color: #FFFFFF !important;
        border: 1px solid #000 !important;
        border-radius: 8px;
        text-decoration: none !important;
        padding: 10px 14px;
        font-weight: 600;
    }
    [data-testid="stPageLink"] a *,
    [data-testid="stPageLink"] a p,
    [data-testid="stPageLink"] a span { color: #FFFFFF !important; }
    [data-testid="stPageLink"] a:hover { background-color: #294A8B !important; }
    [data-testid="stLinkButton"] a {
        border: 1px solid #000;
        border-radius: 7px;
        background-color: #FFFFFF;
        color: #1E376C;
        font-size: .85rem;
        font-weight: 600;
        min-height: 35px;
    }
    [data-testid="stLinkButton"] a:hover { background-color: #EDF2FA; color: #1E376C; }
    .home-description { color: #52647D; font-size: .94rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


def verificar_api():
    try:
        respuesta = requests.get(f"{API_URL}/health", timeout=3)
        respuesta.raise_for_status()
        return respuesta.json().get("status") == "ok"
    except (requests.exceptions.RequestException, ValueError):
        return False


def mostrar_integrante(nombre, usuario, enlace):
    with st.container(border=True):
        col_logo, col_info = st.columns([1, 5], vertical_alignment="center", gap="small")
        with col_logo:
            if GITHUB_LOGO.exists():
                st.image(str(GITHUB_LOGO), width=30)
            else:
                st.markdown("**GitHub**")
        with col_info:
            st.markdown(f"**{nombre}**")
            st.caption(f"@{usuario}")
        st.link_button("Ver perfil de GitHub", url=enlace, use_container_width=True)


mostrar_encabezado("VectorIA", "Asistente Inteligente para la Vigilancia del Dengue en Costa Rica")
st.markdown(
    '<p class="home-description">'
    'Plataforma académica de inteligencia artificial para identificar mosquitos, '
    'pronosticar casos de dengue y generar recomendaciones preventivas.'
    '</p>',
    unsafe_allow_html=True
)
st.divider()

st.subheader("Estado del sistema")
col_api, col_cnn, col_lstm, col_ann = st.columns([1.4, 1, 1, 1], gap="small")
with col_api:
    with st.container(border=True):
        st.markdown("**API · FastAPI**")
        if verificar_api():
            st.success("Conectada")
        else:
            st.error("Desconectada")
with col_cnn:
    with st.container(border=True):
        st.markdown("**CNN**")
        st.caption("Identificación de mosquitos")
with col_lstm:
    with st.container(border=True):
        st.markdown("**LSTM**")
        st.caption("Predicción de casos")
with col_ann:
    with st.container(border=True):
        st.markdown("**ANN**")
        st.caption("Acciones preventivas")
st.caption("La conexión de la API no garantiza que los modelos entrenados estén disponibles.")
st.divider()

st.subheader("Explorar la plataforma")
st.caption("Acceda a las herramientas y visualizaciones de VectorIA.")
col1, col2, col3 = st.columns(3, gap="small")
with col1:
    with st.container(border=True):
        st.markdown("#### 📊 Métricas")
        st.caption("Evaluación, resultados y comparación del rendimiento de los modelos.")
        st.button("Próximamente", disabled=True, use_container_width=True, key="metricas_inicio")
with col2:
    with st.container(border=True):
        st.markdown("#### 🔬 Laboratorio")
        st.caption("Pruebe CNN, LSTM y ANN con datos y obtenga resultados.")
        st.page_link("pages/2_Probar_Modelos.py", label="Abrir laboratorio", icon="🧪", use_container_width=True)
with col3:
    with st.container(border=True):
        st.markdown("#### 🗺️ Mapa de calor")
        st.caption("Explore los casos de dengue por cantón y el bonus provincial.")
        st.button("Próximamente", disabled=True, use_container_width=True, key="mapa_inicio")
st.divider()

st.subheader("Equipo de desarrollo")
st.caption("Integrantes del proyecto VectorIA · Colegio Universitario de Cartago")
integrantes = [
    {"nombre": "Andrey", "usuario": "andreyac1397", "enlace": "https://github.com/andreyac1397"},
    {"nombre": "Mariana", "usuario": "hymari093-cmy", "enlace": "https://github.com/hymari093-cmy"},
    {"nombre": "Mauricio", "usuario": "MauOrtizC", "enlace": "https://github.com/MauOrtizC"},
    {"nombre": "Kaleb", "usuario": "EladrelFG", "enlace": "https://github.com/EladrelFG"},
]
for columna, integrante in zip(st.columns(4, gap="small"), integrantes):
    with columna:
        mostrar_integrante(integrante["nombre"], integrante["usuario"], integrante["enlace"])

st.divider()
st.markdown("**Acerca de VectorIA**")
st.caption("Proyecto académico de Inteligencia Artificial desarrollado en el Colegio Universitario de Cartago.")
st.info(
    "VectorIA es una herramienta experimental de apoyo a la vigilancia epidemiológica. "
    "No diagnostica dengue en personas ni sustituye las decisiones de profesionales de salud."
)
st.caption("VectorIA · Colegio Universitario de Cartago")
