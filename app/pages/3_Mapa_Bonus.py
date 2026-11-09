"""Mapa histórico cantonal de dengue de VectorIA y espacio para bonus provincial."""

import json
import os
from datetime import date

import requests
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
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

# ============================================================
# 5. BONUS PROVINCIAL (INDEPENDIENTE DEL MAPA CANTONAL)
# ============================================================

API_PROVINCIAL_URL = os.getenv("VECTORIA_API_URL", "http://127.0.0.1:8000").rstrip("/")
PROVINCIAS_BONUS = [
    "Alajuela", "Cartago", "Guanacaste", "Heredia",
    "Limón", "Puntarenas", "San José",
]
COLORES_BONUS = [
    [255, 245, 204, 225], [255, 211, 132, 230],
    [254, 166, 75, 235], [246, 106, 54, 240],
    [204, 45, 52, 245], [128, 22, 51, 250],
]


def _nombre_provincia(nombre):
    # Homologación mínima entre el GeoJSON cantonal y el modelo.
    import unicodedata
    return "".join(
        letra for letra in unicodedata.normalize("NFD", str(nombre).strip().lower())
        if unicodedata.category(letra) != "Mn"
    )


@st.cache_data(show_spinner=False)
def geojson_provincial_desde_cantones(archivo_geojson):
    """Agrupa los mismos polígonos cantonales por provincia; no usa datos históricos."""
    agrupados = {p: [] for p in PROVINCIAS_BONUS}
    nombres = {_nombre_provincia(p): p for p in PROVINCIAS_BONUS}
    for feature in archivo_geojson["features"]:
        original = feature["properties"]["NAME_1"]
        provincia = nombres.get(_nombre_provincia(original))
        if provincia:
            agrupados[provincia].append(shape(feature["geometry"]))
    sin_geometria = [p for p, geoms in agrupados.items() if not geoms]
    if sin_geometria:
        raise ValueError(f"No se encontraron cantones para: {sin_geometria}")
    provincias_geo = []
    puntos = {}
    for provincia, geometrias in agrupados.items():
        poligono = unary_union(geometrias)
        punto = poligono.representative_point()
        puntos[provincia] = [float(punto.x), float(punto.y)]
        provincias_geo.append({
            "type": "Feature",
            "properties": {"provincia": provincia},
            "geometry": mapping(poligono),
        })
    return {"type": "FeatureCollection", "features": provincias_geo}, puntos


def consultar_prediccion_provincial(fecha, provincia):
    try:
        respuesta = requests.post(
            f"{API_PROVINCIAL_URL}/lstm-provincial/predecir-dengue",
            json={
                "fecha": fecha.isoformat(),
                "provincia": None,  # El mapa necesita las siete provincias; la UI filtra después.
            },
            timeout=120,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()
        if datos.get("estado") != "correcto" or not isinstance(datos.get("resultados"), list):
            raise ValueError("El servicio provincial no devolvió resultados válidos.")
        return datos
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con FastAPI. Verificá que esté encendido en el puerto 8000.")
    except requests.exceptions.Timeout:
        st.error("La predicción provincial superó el tiempo de espera de la API.")
    except requests.exceptions.HTTPError as error:
        try:
            detalle = error.response.json().get("detail", error.response.text)
        except ValueError:
            detalle = error.response.text
        st.error(f"Error de FastAPI ({error.response.status_code}): {detalle}")
    except (ValueError, requests.exceptions.RequestException) as error:
        st.error(f"No se pudo leer la predicción provincial: {error}")
    return None


def _validar_resultados_provinciales(resultado):
    filas = []
    for item in resultado["resultados"]:
        provincia = item.get("provincia")
        predicciones = item.get("predicciones")
        if provincia not in PROVINCIAS_BONUS or not isinstance(predicciones, list) or len(predicciones) != 4:
            raise ValueError("La API envió provincias o semanas inesperadas.")
        for pred in predicciones:
            casos = pred.get("casos_estimados")
            if (isinstance(casos, bool) or not isinstance(casos, (int, float))
                    or not math.isfinite(casos) or casos < 0):
                raise ValueError("La API envió una cantidad de casos inválida.")
            filas.append({
                "Provincia": provincia,
                "Semana": int(pred["semana"]),
                "Fecha": pd.to_datetime(pred["fecha"]).date(),
                "Casos estimados": int(round(casos)),
            })
    if not filas or pd.DataFrame(filas).duplicated(["Provincia", "Fecha"]).any():
        raise ValueError("La respuesta provincial está vacía o contiene semanas duplicadas.")
    return pd.DataFrame(filas)


def mapa_provincial_2d(provincias_geo, totales, seleccion):
    valores = totales[totales["Provincia"] == seleccion] if seleccion != "Todas" else totales
    resto = [p for p in PROVINCIAS_BONUS if p not in valores["Provincia"].tolist()]
    fig = go.Figure()
    if resto:
        fig.add_trace(go.Choroplethmap(
            geojson=provincias_geo, locations=resto,
            z=[1] * len(resto), featureidkey="properties.provincia",
            colorscale=[[0, "#BCC6D2"], [1, "#BCC6D2"]],
            showscale=False, marker_opacity=0.65,
            marker_line_color="#344054", marker_line_width=1.0,
            text=[f"{p}<br>No seleccionada" for p in resto],
            hovertemplate="%{text}<extra></extra>",
        ))
    if not valores.empty:
        maximo = max(float(totales["Casos estimados"].max()), 1.0)
        fig.add_trace(go.Choroplethmap(
            geojson=provincias_geo,
            locations=valores["Provincia"],
            z=valores["Casos estimados"],
            featureidkey="properties.provincia",
            zmin=0, zmax=maximo, colorscale="YlOrRd",
            colorbar={"title": "Casos (4 sem.)", "thickness": 15, "len": 0.75},
            marker_opacity=0.88, marker_line_color="#344054", marker_line_width=1.0,
            text=[
                f"<b>{p}</b><br>Casos estimados (4 semanas): {int(c):,}"
                for p, c in zip(valores["Provincia"], valores["Casos estimados"])
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(
        map={"style": "carto-positron", "center": {"lat": 9.75, "lon": -84.15}, "zoom": 6.7},
        height=590, margin={"l": 0, "r": 0, "t": 5, "b": 0},
        paper_bgcolor="#FFFFFF", showlegend=False, font={"color": "#17243B"},
    )
    return fig


def mapa_provincial_hexagonal(provincias_geo, puntos, totales, seleccion, radio_km, elevacion):
    maximo = max(float(totales["Casos estimados"].max()), 1.0)
    por_provincia = dict(zip(totales["Provincia"], totales["Casos estimados"]))
    hexagonos = []
    for provincia in PROVINCIAS_BONUS:
        longitud, latitud = puntos[provincia]
        radio_metros = radio_km * 1000 * 0.5  # Mismo criterio del mapa cantonal.
        delta_lat = radio_metros / 111_320
        delta_lon = radio_metros / (111_320 * max(math.cos(math.radians(latitud)), 0.1))
        vertices = [
            [longitud + delta_lon * math.cos(math.radians(60 * i + 30)),
             latitud + delta_lat * math.sin(math.radians(60 * i + 30))]
            for i in range(6)
        ]
        casos = por_provincia[provincia]
        if seleccion != "Todas" and provincia != seleccion:
            color = [170, 180, 192, 180]
            altura = 70
            texto = "No seleccionada"
        elif casos == 0:
            color = COLORES_BONUS[0]
            altura = 70
            texto = "0"
        else:
            proporcion = math.log1p(casos) / math.log1p(maximo)
            color = COLORES_BONUS[min(5, max(0, int(proporcion * 5.999)))]
            altura = 20000 * elevacion * proporcion
            texto = f"{casos:,}"
        hexagonos.append({
            "polygon": vertices, "fill_color": color,
            "altura": altura, "casos_texto": texto, "provincia": provincia,
        })
    return pdk.Deck(
        layers=[
            pdk.Layer("GeoJsonLayer", data=provincias_geo, filled=False, stroked=True,
                      get_line_color=[62, 77, 104, 140], get_line_width=300,
                      line_width_min_pixels=0.5, pickable=False),
            pdk.Layer("PolygonLayer", data=hexagonos, get_polygon="polygon",
                      get_fill_color="fill_color", get_elevation="altura",
                      extruded=True, filled=True, stroked=True, wireframe=False,
                      get_line_color=[65, 65, 70, 180], line_width_min_pixels=1,
                      pickable=True, auto_highlight=True, opacity=0.95),
        ],
        initial_view_state=pdk.ViewState(
            latitude=9.74, longitude=-84.13, zoom=6.7, pitch=48, bearing=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": "<b>{provincia}</b><br/>Casos estimados (4 semanas): {casos_texto}",
            "style": {"backgroundColor": "#1E376C", "color": "white"},
        },
    )


with tab_bonus:
    st.subheader("Predicción provincial de dengue · Bonus LSTM")
    st.caption(
        "Pronóstico experimental de cuatro semanas sobre datos provinciales sintéticos "
        "(2014–2025). Independiente del LSTM nacional y del mapa de calor cantonal."
    )
    if "bonus_provincial_resultado" not in st.session_state:
        st.session_state["bonus_provincial_resultado"] = None
    if "bonus_provincial_consulta" not in st.session_state:
        st.session_state["bonus_provincial_consulta"] = None

    c1, c2, c3 = st.columns([1.2, 1.2, 1], gap="medium")
    with c1:
        provincia_bonus = st.selectbox(
            "Provincia", ["Todas"] + PROVINCIAS_BONUS,
            key="bonus_provincia_elegida",
        )
    with c2:
        fecha_bonus = st.date_input(
            "Fecha de referencia (2026)", value=date(2026, 1, 4),
            min_value=date(2026, 1, 1), max_value=date(2026, 12, 27),
            format="DD/MM/YYYY", key="bonus_fecha_elegida",
        )
    with c3:
        st.write("")
        st.write("")
        generar_bonus = st.button(
            "Generar predicción", type="primary", use_container_width=True,
            key="bonus_generar_prediccion",
        )

    consulta_actual = (str(fecha_bonus), provincia_bonus)
    if generar_bonus:
        st.session_state["bonus_provincial_resultado"] = None
        st.session_state["bonus_provincial_consulta"] = None
        with st.spinner("Consultando LSTM provincial..."):
            respuesta = consultar_prediccion_provincial(fecha_bonus, provincia_bonus)
        if respuesta is not None:
            st.session_state["bonus_provincial_resultado"] = respuesta
            st.session_state["bonus_provincial_consulta"] = consulta_actual

    resultado_bonus = st.session_state["bonus_provincial_resultado"]
    if resultado_bonus is None:
        st.info("Elegí una provincia y una fecha de 2026, y presioná Generar predicción.")
    elif st.session_state["bonus_provincial_consulta"] != consulta_actual:
        st.info("Cambiaste los parámetros. Presioná Generar predicción para actualizar el mapa.")
    else:
        try:
            predicciones_bonus = _validar_resultados_provinciales(resultado_bonus)
            provincias_geo, puntos_geo = geojson_provincial_desde_cantones(geojson)
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"No se pudieron preparar los resultados provinciales: {error}")
        else:
            totales_bonus = (
                predicciones_bonus.groupby("Provincia", as_index=False)["Casos estimados"]
                .sum()
            )
            total_predicho = int(
                totales_bonus["Casos estimados"].sum()
                if provincia_bonus == "Todas"
                else totales_bonus.loc[
                    totales_bonus["Provincia"] == provincia_bonus, "Casos estimados"
                ].sum()
            )
            st.metric(
                "Casos estimados en 4 semanas" if provincia_bonus != "Todas"
                else "Casos estimados en las 7 provincias (4 semanas)",
                f"{total_predicho:,}",
            )
            st.caption(
                f"Semanas pronosticadas: {resultado_bonus['primera_semana']} "
                f"y tres semanas posteriores. El modelo toma el domingo coincidente "
                f"o siguiente a la fecha seleccionada."
            )

            vista_bonus = st.radio(
                "Tipo de mapa provincial",
                ["Hexágonos 3D", "Provincias 2D"],
                horizontal=True, key="bonus_tipo_mapa",
            )
            with st.container(border=True):
                if vista_bonus == "Hexágonos 3D":
                    br, be = st.columns(2, gap="medium")
                    with br:
                        radio_bonus = st.slider(
                            "Radio de hexágonos (km)", 8, 45, 19, step=1,
                            key="bonus_radio_hexagonos",
                        )
                    with be:
                        elevacion_bonus = st.slider(
                            "Escala de altura", 0.1, 2.0, 0.65, step=0.05,
                            key="bonus_altura_hexagonos",
                        )
                    col_mapa, col_leyenda = st.columns([11, 2], gap="small")
                    with col_mapa:
                        st.pydeck_chart(
                            mapa_provincial_hexagonal(
                                provincias_geo, puntos_geo, totales_bonus,
                                provincia_bonus, radio_bonus, elevacion_bonus,
                            ),
                            use_container_width=True, height=610,
                        )
                    with col_leyenda:
                        mostrar_leyenda_hexagonal(
                            totales_bonus.rename(columns={"Casos estimados": "casos_dengue"})
                        )
                    st.caption(
                        "Cada hexágono representa una provincia, no un cantón ni un lugar "
                        "de contagio. Gris = provincia no seleccionada. Gire con Shift."
                    )
                else:
                    st.plotly_chart(
                        mapa_provincial_2d(provincias_geo, totales_bonus, provincia_bonus),
                        use_container_width=True,
                        config={"displaylogo": False, "scrollZoom": True},
                    )
                    st.caption(
                        "Los polígonos provinciales se construyen agrupando los cantones "
                        "del mismo GeoJSON existente. Gris = provincia no seleccionada."
                    )

            st.markdown("#### Detalle de predicciones · próximas cuatro semanas")
            detalle_bonus = predicciones_bonus.copy()
            if provincia_bonus != "Todas":
                detalle_bonus = detalle_bonus[detalle_bonus["Provincia"] == provincia_bonus]
            detalle_bonus = detalle_bonus.sort_values(["Provincia", "Fecha"])
            st.dataframe(
                detalle_bonus, hide_index=True, use_container_width=True,
                column_config={
                    "Fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                    "Casos estimados": st.column_config.NumberColumn(format="%d"),
                },
            )
            if provincia_bonus == "Todas":
                grafico_bonus = (
                    predicciones_bonus.groupby("Fecha", as_index=False)["Casos estimados"]
                    .sum()
                )
                titulo_grafico = "Total de casos estimados por semana · 7 provincias"
            else:
                grafico_bonus = detalle_bonus[["Fecha", "Casos estimados"]].copy()
                titulo_grafico = f"Casos estimados por semana · {provincia_bonus}"
            st.markdown(f"#### {titulo_grafico}")
            st.line_chart(grafico_bonus.set_index("Fecha"), use_container_width=True)
            st.download_button(
                "Descargar predicciones (CSV)",
                data=detalle_bonus.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"vectoria_prediccion_provincial_{fecha_bonus.isoformat()}.csv",
                mime="text/csv", key="bonus_descargar_csv",
            )
            st.warning(resultado_bonus.get("advertencia", "Datos sintéticos; no son pronósticos oficiales."))
