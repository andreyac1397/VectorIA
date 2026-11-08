"""Mapa histórico cantonal de dengue de VectorIA y espacio para bonus provincial."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
from shapely.geometry import shape
import streamlit as st

# ============================================================
# 1. CONFIGURACIÓN Y TEMA COMPARTIDO
# ============================================================

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.theme import aplicar_tema, mostrar_identidad, mostrar_encabezado

GEOJSON_PATH = (
    PROJECT_DIR / "data" / "processed" / "mapa_calor"
    / "vectoria_mapa_cantonal.geojson"
)
CSV_PATH = (
    PROJECT_DIR / "data" / "processed" / "mapa_calor"
    / "vectoria_casos_cantonal_limpio.csv"
)
ANIOS = list(range(2013, 2019))

st.set_page_config(
    page_title="Mapa de calor | VectorIA",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

aplicar_tema()
mostrar_identidad()
mostrar_encabezado(
    "Mapa de calor y bonus",
    "Distribución histórica del dengue por cantón en Costa Rica (2013–2018).",
)


# ============================================================
# 2. CARGA Y VALIDACIÓN DE ARCHIVOS
# ============================================================

@st.cache_data(show_spinner=False)
def cargar_datos():
    with GEOJSON_PATH.open("r", encoding="utf-8") as archivo:
        geojson = json.load(archivo)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError("El archivo no es una colección GeoJSON válida.")

    features = geojson.get("features", [])
    if not features:
        raise ValueError("El GeoJSON no contiene cantones.")

    for feature in features:
        props = feature.get("properties", {})
        if not all(campo in props for campo in ("ISO", "NAME_1", "NAME_2")):
            raise ValueError("Faltan identificadores geográficos en el GeoJSON.")
        if not all(f"casos_{anio}" in props for anio in ANIOS):
            raise ValueError("El GeoJSON no incluye casos para todos los años.")

    codigos = [f["properties"]["ISO"] for f in features]
    if len(codigos) != len(set(codigos)):
        raise ValueError("Existen códigos ISO duplicados en el GeoJSON.")

    csv = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    requeridas = {"anio", "provincia", "canton", "casos_dengue", "estado"}
    if not requeridas.issubset(csv.columns):
        raise ValueError("El CSV procesado no tiene las columnas esperadas.")

    return geojson, csv


try:
    geojson, datos_csv = cargar_datos()
except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
    st.error("No se pudieron cargar los archivos procesados del mapa.")
    st.code(str(error))
    st.info(
        "Compruebe que los archivos vectoria_mapa_cantonal.geojson y "
        "vectoria_casos_cantonal_limpio.csv estén en "
        "data/processed/mapa_calor/."
    )
    st.stop()


# ============================================================
# 3. FUNCIONES DEL MAPA
# ============================================================

def preparar_anio(archivo_geojson, anio):
    filas = []
    for feature in archivo_geojson["features"]:
        p = feature["properties"]
        casos = p.get(f"casos_{anio}")
        filas.append(
            {
                "codigo": p["ISO"],
                "provincia": p["NAME_1"],
                "canton": p["NAME_2"],
                "casos_dengue": casos,
                "estado": "Sin datos" if casos is None else "Con datos",
            }
        )

    df = pd.DataFrame(filas)
    total = df["casos_dengue"].sum()
    df["participacion"] = (
        df["casos_dengue"] / total * 100 if total > 0 else 0.0
    )
    df.loc[df["casos_dengue"].isna(), "participacion"] = float("nan")
    return df


def construir_mapa(archivo_geojson, df, variable):
    disponibles = df[df["casos_dengue"].notna()].copy()
    faltantes = df[df["casos_dengue"].isna()].copy()
    figura = go.Figure()

    # Base gris: exclusivamente los cantones sin registros.
    if not faltantes.empty:
        figura.add_trace(
            go.Choroplethmap(
                geojson=archivo_geojson,
                locations=faltantes["codigo"],
                z=[1] * len(faltantes),
                featureidkey="properties.ISO",
                colorscale=[[0, "#B9C2CE"], [1, "#B9C2CE"]],
                showscale=False,
                marker_opacity=0.78,
                marker_line_color="#344054",
                marker_line_width=0.7,
                text=[
                    f"<b>{r.canton}</b><br>Provincia: {r.provincia}<br>Sin datos"
                    for r in faltantes.itertuples()
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Sin datos",
            )
        )

    # Valores disponibles: los ceros son datos válidos.
    if not disponibles.empty:
        columna = "casos_dengue" if variable == "Casos registrados" else "participacion"
        titulo_escala = "Casos" if variable == "Casos registrados" else "Participación (%)"
        valores = disponibles[columna].astype(float)
        maximo = max(float(valores.max()), 1.0)

        figura.add_trace(
            go.Choroplethmap(
                geojson=archivo_geojson,
                locations=disponibles["codigo"],
                z=valores,
                zmin=0,
                zmax=maximo,
                featureidkey="properties.ISO",
                colorscale="YlOrRd",
                colorbar={"title": titulo_escala, "thickness": 15, "len": 0.75},
                marker_opacity=0.85,
                marker_line_color="#344054",
                marker_line_width=0.7,
                text=[
                    (
                        f"<b>{r.canton}</b><br>Provincia: {r.provincia}"
                        f"<br>Casos: {int(r.casos_dengue):,}"
                        f"<br>Participación anual: {r.participacion:.2f}%"
                    )
                    for r in disponibles.itertuples()
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Con datos",
            )
        )

    figura.update_layout(
        map={
            "style": "carto-positron",
            "center": {"lat": 9.75, "lon": -84.15},
            "zoom": 6.7,
        },
        height=610,
        margin={"l": 0, "r": 0, "t": 5, "b": 0},
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        font={"color": "#17243B"},
    )
    return figura


@st.cache_data(show_spinner=False)
def preparar_hexagonos_3d(archivo_geojson, anio, radio_metros, escala_altura):
    """Un prisma hexagonal por cantón, no ubicaciones de pacientes.

    PolygonLayer es una alternativa estable cuando HexagonLayer no renderiza
    en el navegador. El mapa no agrega cantones por celda espacial.
    """
    registros = []
    for feature in archivo_geojson["features"]:
        propiedades = feature["properties"]
        casos = propiedades.get(f"casos_{anio}")
        punto = shape(feature["geometry"]).representative_point()
        registros.append({
            "lon": float(punto.x),
            "lat": float(punto.y),
            "casos": None if casos is None else int(casos),
            "canton": propiedades["NAME_2"],
            "provincia": propiedades["NAME_1"],
        })

    positivos = [r["casos"] for r in registros if r["casos"] is not None]
    maximo = max(positivos, default=1)
    maximo = max(maximo, 1)
    niveles = [
        [255, 245, 204, 225],
        [255, 211, 132, 230],
        [254, 166, 75, 235],
        [246, 106, 54, 240],
        [204, 45, 52, 245],
        [128, 22, 51, 250],
    ]

    hexagonos = []
    for fila in registros:
        latitud = fila["lat"]
        longitud = fila["lon"]
        # Radio en metros, convertido a grados para este punto geográfico.
        delta_lat = radio_metros / 111_320
        delta_lon = radio_metros / (111_320 * max(math.cos(math.radians(latitud)), 0.1))
        vertices = [
            [
                longitud + delta_lon * math.cos(math.radians(60 * i + 30)),
                latitud + delta_lat * math.sin(math.radians(60 * i + 30)),
            ]
            for i in range(6)
        ]

        casos = fila["casos"]
        if casos is None:
            color = [170, 180, 192, 190]
            altura = 70
            etiqueta = "Sin datos"
        elif casos == 0:
            color = niveles[0]
            altura = 70
            etiqueta = "0"
        else:
            proporcion = math.log1p(casos) / math.log1p(maximo)
            indice_color = min(5, max(0, int(proporcion * 5.999)))
            color = niveles[indice_color]
            # Altura visual normalizada para evitar torres desproporcionadas.
            altura = 20000 * escala_altura * proporcion
            etiqueta = f"{casos:,}"

        hexagonos.append({
            "polygon": vertices,
            "fill_color": color,
            "altura": altura,
            "casos_texto": etiqueta,
            "canton": fila["canton"],
            "provincia": fila["provincia"],
        })
    return hexagonos


def construir_mapa_hexagonal(archivo_geojson, anio, radio, elevacion):
    """Prismas hexagonales 3D construidos con deck.gl PolygonLayer."""
    # Reducir tamaño del radio elegido para evitar demasiado solapamiento.
    hexagonos = preparar_hexagonos_3d(archivo_geojson, anio, radio * 0.5, elevacion)
    if not hexagonos:
        return None

    capa_limites = pdk.Layer(
        "GeoJsonLayer",
        data=archivo_geojson,
        filled=False,
        stroked=True,
        get_line_color=[62, 77, 104, 110],
        get_line_width=300,
        line_width_min_pixels=0.5,
        pickable=False,
    )
    capa_hexagonos = pdk.Layer(
        "PolygonLayer",
        data=hexagonos,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_elevation="altura",
        extruded=True,
        filled=True,
        stroked=True,
        wireframe=False,
        get_line_color=[65, 65, 70, 180],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        opacity=0.95,
    )

    return pdk.Deck(
        layers=[capa_limites, capa_hexagonos],
        initial_view_state=pdk.ViewState(
            latitude=9.74,
            longitude=-84.13,
            zoom=6.7,
            pitch=48,
            bearing=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": "<b>{canton}, {provincia}</b><br/>Casos: {casos_texto}",
            "style": {"backgroundColor": "#1E376C", "color": "white"},
        },
    )



def mostrar_leyenda_hexagonal(df):
    """Barra vertical al lado del mapa, con los colores y la escala del 3D."""
    import html

    colores = ["#FFF5CC", "#FFD384", "#FEA64B", "#F66A36", "#CC2D34", "#801633"]
    valores = df["casos_dengue"].dropna().astype(float)
    maximo = max(float(valores.max()), 1.0) if not valores.empty else 1.0

    # Los prismas usan seis intervalos de log1p(casos)/log1p(maximo).
    # La barra repite esos intervalos, ordenados del mayor al menor.
    pasos = []
    for indice, color in enumerate(colores):
        desde = indice / 6 * 100
        hasta = (indice + 1) / 6 * 100
        pasos.extend([f"{color} {desde:.4f}%", f"{color} {hasta:.4f}%"])
    gradiente = "linear-gradient(to top, " + ", ".join(pasos) + ")"

    def etiqueta(valor):
        return f"{valor:,.0f}" if maximo >= 100 else f"{valor:,.1f}"

    etiquetas = []
    for fraccion in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        valor = math.expm1(fraccion * math.log1p(maximo))
        etiquetas.append(
            f'<span style="position:absolute;top:{(1-fraccion)*100:.2f}%;'
            'transform:translateY(-50%);left:0;white-space:nowrap;'
            'font-size:12px;color:#344054">'
            + html.escape(etiqueta(valor)) + '</span>'
        )

    st.markdown(
        '<div style="padding:30px 3px 0 0;">'
        '<div style="font-size:13px;font-weight:600;color:#17243B;'
        'margin-bottom:15px;">Casos</div>'
        '<div style="display:flex;gap:9px;align-items:stretch;">'
        f'<div style="width:20px;height:470px;flex-shrink:0;background:{gradiente};'
        'border:1px solid #D6DCE5;"></div>'
        '<div style="position:relative;height:470px;flex:1;min-width:48px;">'
        + "".join(etiquetas) + '</div></div>'
        '<div style="display:flex;align-items:center;gap:6px;'
        'margin-top:22px;font-size:12px;color:#344054;">'
        '<span style="width:16px;height:16px;background:#AAB4C0;'
        'border:1px solid #8D99A7;display:inline-block;"></span>'
        'Sin datos</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 4. PESTAÑAS
# ============================================================

tab_mapa, tab_bonus = st.tabs(["🗺️ Mapa cantonal", "📈 Bonus provincial"])

with tab_mapa:
    st.subheader("Distribución histórica de casos de dengue")
    st.caption(
        "Seleccione un año para consultar los casos registrados en el dataset "
        "por cantón. Los territorios sin información aparecen en gris."
    )

    col_anio, col_metrica = st.columns([1, 2], gap="medium")
    with col_anio:
        anio = st.selectbox("Año", ANIOS, index=0)
    with col_metrica:
        vista = st.radio(
            "Tipo de mapa",
            ["Hexágonos 3D", "Cantones 2D"],
            horizontal=True,
        )
        variable = "Casos registrados"
        if vista == "Cantones 2D":
            variable = st.radio(
                "Colorear según",
                ["Casos registrados", "Participación anual (%)"],
                horizontal=True,
            )

    df_anio = preparar_anio(geojson, anio)
    con_datos = int(df_anio["casos_dengue"].notna().sum())
    sin_datos = int(df_anio["casos_dengue"].isna().sum())
    total = int(df_anio["casos_dengue"].sum())
    mas_casos = df_anio.loc[df_anio["casos_dengue"].idxmax()]

    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        st.metric("Cantones con datos", con_datos)
    with m2:
        st.metric("Cantones sin datos", sin_datos)
    with m3:
        st.metric("Total de casos registrados del año", f"{total:,}")
    with m4:
        st.metric("Cantón con más casos", mas_casos["canton"])

    if vista == "Hexágonos 3D":
        col_radio, col_altura = st.columns(2, gap="medium")
        with col_radio:
            radio = st.slider("Radio de hexágonos (km)", 8, 45, 19, step=1)
        with col_altura:
            elevacion = st.slider("Escala de altura", 0.1, 2.0, 0.65, step=0.05)

    with st.container(border=True):
        if vista == "Hexágonos 3D":
            mapa3d = construir_mapa_hexagonal(geojson, anio, radio * 1000, elevacion)
            if mapa3d is None:
                st.warning("No hay registros para visualizar en este año.")
            else:
                col_mapa_3d, col_leyenda_3d = st.columns([11, 2], gap="small")
                with col_mapa_3d:
                    st.pydeck_chart(mapa3d, use_container_width=True, height=610)
                with col_leyenda_3d:
                    mostrar_leyenda_hexagonal(df_anio)
            st.caption(
                "Visualización 3D exploratoria: cada prisma representa un cantón, "
                "ubicado en un punto representativo dentro de su polígono; no señala "
                "lugares reales de contagio. Gris = sin datos; amarillo claro = cero "
                "casos registrados. Las alturas están normalizadas para visualización. "
                "Gire el mapa manteniendo Shift presionado."
            )
        else:
            st.plotly_chart(
                construir_mapa(geojson, df_anio, variable),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
            )
            st.caption(
                "Gris = sin registro; un valor de 0 representa un registro válido. "
                "Puede acercar el mapa y pasar el cursor sobre cada cantón."
            )

    st.subheader(f"Detalle de cantones · {anio}")
    provincias = sorted(df_anio["provincia"].unique())
    seleccion = st.selectbox("Filtrar por provincia", ["Todas"] + provincias)
    tabla = df_anio.copy()
    if seleccion != "Todas":
        tabla = tabla[tabla["provincia"] == seleccion]
    tabla = tabla.sort_values("casos_dengue", ascending=False, na_position="last")
    tabla = tabla.rename(
        columns={
            "provincia": "Provincia",
            "canton": "Cantón",
            "casos_dengue": "Casos registrados",
            "participacion": "Participación (%)",
            "estado": "Estado",
        }
    )
    tabla = tabla[
        ["Provincia", "Cantón", "Casos registrados", "Participación (%)", "Estado"]
    ]
    st.dataframe(
        tabla,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Casos registrados": st.column_config.NumberColumn(format="%d"),
            "Participación (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.download_button(
        "Descargar tabla del año (CSV)",
        data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"vectoria_cantones_{anio}.csv",
        mime="text/csv",
    )

    st.info(
        "**Nota metodológica:** los totales cantonales de este archivo no equivalen "
        "necesariamente a los totales nacionales oficiales. Los valores representan "
        "casos registrados en el dataset; sin datos no significa cero casos. "
        "Sin población cantonal no es posible calcular tasas de incidencia."
    )

with tab_bonus:
    st.subheader("Predicción provincial · Bonus")
    st.info(
        "Este módulo está pendiente de integrar. Utilizará el dataset semanal "
        "provincial y su modelo predictivo correspondiente; no se generarán "
        "predicciones provinciales a partir del mapa cantonal histórico."
    )
