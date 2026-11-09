
from pathlib import Path
from html import escape

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
LOGO_CUC = APP_DIR / "assets" / "logo_cuc.png"


def aplicar_tema():
    """Aplica el estilo institucional común a todas las páginas."""

    st.markdown(
        """
        <style>
        /* ==========================================
           1. DISEÑO GENERAL
        ========================================== */

        .stApp {
            background-color: #F3F6FA;
            color: #17243B;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            color: #1E376C;
            letter-spacing: -0.4px;
        }


        /* ==========================================
           2. MENÚ LATERAL
        ========================================== */

        [data-testid="stSidebar"] {
            min-width: 250px !important;
            max-width: 250px !important;
            background-color: #1E376C !important;
            border-right: 1px solid #000000;
        }

        [data-testid="stSidebar"] * {
            color: #FFFFFF;
        }

        [data-testid="stSidebarNav"] {
            padding-top: 12px;
        }

        [data-testid="stSidebarNav"] a {
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 6px;
            transition: background-color 0.2s ease;
        }

        [data-testid="stSidebarNav"] a:hover {
            background-color: #344D7D;
        }

        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background-color: #344D7D;
            border-left: 3px solid #EF4135;
        }


        /* ==========================================
           3. IDENTIDAD INSTITUCIONAL
        ========================================== */

        [data-testid="stSidebarUserContent"] {
            position: absolute;
            bottom: 16px;
            left: 0;
            width: 100%;
            padding: 0 20px;
            background: transparent !important;
        }

        [data-testid="stSidebarUserContent"] > div {
            background: transparent !important;
        }

        [data-testid="stSidebarUserContent"]
        [data-testid="stMarkdown"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebarUserContent"] h3 {
            color: #FFFFFF !important;
            font-size: 22px;
            margin: 0 0 8px;
        }

        [data-testid="stSidebarUserContent"] p {
            color: #D5E0EF !important;
            font-size: 13px;
            line-height: 1.45;
            margin: 0 0 9px;
        }

        [data-testid="stSidebarUserContent"] strong {
            color: #FFFFFF !important;
        }


        /* ==========================================
           4. CONTENEDORES PRINCIPALES
           Bordes negros en todas las páginas
        ========================================== */

        [data-testid="stMain"]
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #000000 !important;
            border-radius: 10px !important;
        }

        [data-testid="stMain"]
        [data-testid="stVerticalBlockBorderWrapper"]
        > div {
            border-color: #000000 !important;
            border-radius: 10px;
        }

        [data-testid="stMain"]
        [data-testid="stVerticalBlockBorderWrapper"]:
        has([data-testid="stVerticalBlock"]) {
            background-color: #FFFFFF;
        }

        /* Secciones desplegables */
        [data-testid="stMain"] details {
            background-color: #FFFFFF !important;
            border: 1px solid #000000 !important;
            border-radius: 10px !important;
            overflow: hidden;
        }

        [data-testid="stMain"] details summary {
            color: #1E376C;
            font-weight: 600;
        }

        [data-testid="stMain"] details[open] {
            border-color: #000000 !important;
        }


        /* ==========================================
           5. TARJETAS DE MÉTRICAS
        ========================================== */

        [data-testid="stMain"] [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #000000;
            border-radius: 10px;
            padding: 16px;
        }

        [data-testid="stMetricLabel"] {
            color: #52647D;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #1E376C;
        }


        /* ==========================================
           6. CAMPOS DE FORMULARIOS
        ========================================== */

        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input {
            background-color: #FFFFFF;
            border: 1px solid #000000;
            border-radius: 7px;
            color: #17243B;
        }

        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF;
            border: 1px dashed #000000;
            border-radius: 10px;
        }


        /* ==========================================
           7. BOTONES
        ========================================== */

        div.stButton > button {
            border: 1px solid #000000;
            border-radius: 8px;
            font-weight: 600;
        }

        div.stButton > button[kind="primary"] {
            background-color: #1E376C;
            color: #FFFFFF;
            border: 1px solid #000000;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #294A8B;
            color: #FFFFFF;
        }

        [data-testid="stPageLink"] a {
            border: 1px solid #000000;
            border-radius: 8px;
        }


        /* ==========================================
           8. PESTAÑAS
        ========================================== */

        [data-testid="stTabs"] [role="tablist"] {
            gap: 12px;
            border-bottom: 1px solid #000000;
        }

        [data-testid="stTabs"] button[role="tab"] {
            font-weight: 600;
            padding: 12px 22px;
        }

        [data-testid="stTabs"]
        button[aria-selected="true"] {
            color: #1E376C;
            border-bottom: 3px solid #EF4135;
        }


        /* ==========================================
           9. TABLAS, GRÁFICOS Y ALERTAS
        ========================================== */

        [data-testid="stMain"] [data-testid="stAlert"] {
            border: 1px solid #000000;
            border-radius: 9px;
        }

        [data-testid="stMain"] [data-testid="stDataFrame"] {
            border: 1px solid #000000;
            border-radius: 8px;
            overflow: hidden;
        }

        [data-testid="stMain"] [data-testid="stPlotlyChart"] {
            border: 1px solid #000000;
            border-radius: 10px;
            background-color: #FFFFFF;
            overflow: hidden;
        }

        [data-testid="stImage"] img {
            border-radius: 8px;
        }

        hr {
            border-color: #B6C4D5;
        }


        /* ==========================================
           10. CLASES PERSONALIZADAS
        ========================================== */

        .section-description {
            color: #52647D;
            font-size: 0.95rem;
            margin-bottom: 20px;
        }

        .small-label {
            color: #52647D;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .result-title {
            color: #1E376C;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .result-panel {
            background-color: #FFFFFF;
            border: 1px solid #000000;
            border-radius: 10px;
            padding: 18px;
            margin: 12px 0;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def mostrar_identidad():
    """Muestra la identidad institucional al fondo del menú lateral."""

    with st.sidebar:
        st.markdown("### VectorIA")
        st.markdown("Proyecto de Inteligencia Artificial Aplicada")
        st.markdown("**Colegio Universitario de Cartago**")


def mostrar_encabezado(titulo, descripcion=""):
    """Muestra el título a la izquierda y el logo del CUC a la derecha."""

    columna_titulo, columna_logo = st.columns(
        [4, 2],
        gap="medium",
        vertical_alignment="center"
    )

    with columna_titulo:
        st.title(titulo)

        if descripcion:
            st.markdown(
                f'<p class="section-description">'
                f'{escape(descripcion)}'
                f'</p>',
                unsafe_allow_html=True
            )

    with columna_logo:
        if LOGO_CUC.exists():
            st.image(
                str(LOGO_CUC),
                use_container_width=True
            )
        else:
            st.caption("Logo institucional no disponible")
