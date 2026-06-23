import unicodedata
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import altair as alt

from utils import normalize, build_payload
from api import api_get, api_post
from data import DIAS, MESES, TURNOS, cargar_barrio_comuna, cargar_geojson, detectar_clave

st.set_page_config(page_title="Seguridad CABA", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ─────────────────────────── API CALLS ───────────────────────────

@st.cache_data(ttl=24 * 3600)
def feriados_anio(anio: int) -> set[tuple[int, int]]:
    try:
        r = requests.get(f"https://api.argentinadatos.com/v1/feriados/{anio}", timeout=10)
        r.raise_for_status()
        out = set()
        for h in r.json():
            _, m, d = map(int, h["fecha"].split("-"))
            out.add((m, d))
        return out
    except Exception:
        return set()

# ─────────────────────────── LOAD STATIC ───────────────────────────

barrio_comuna = cargar_barrio_comuna()
geojson = cargar_geojson()
clave_barrio_geo = detectar_clave(geojson, ("BARRIO", "barrio", "NOMBRE", "nombre", "Nombre"))

barrios_geojson = [f["properties"][clave_barrio_geo] for f in geojson["features"]]
train_keys = {normalize(b): b for b in barrio_comuna}
barrio_geo_to_train: dict[str, str] = {}
for b in barrios_geojson:
    k = normalize(b)
    if k in train_keys:
        barrio_geo_to_train[b] = train_keys[k]

# ─────────────────────────── LOAD API METADATA ───────────────────────────
with st.spinner("Conectando con la API (puede tardar ~30s si estaba dormida)…"):
    metadata = api_get("/metadata")

umbral = metadata["umbral"]
anios_entrenamiento = metadata["anios_entrenamiento"]
anios_disponibles = metadata["anios_disponibles"]
modelo_es_hibrido = metadata["modelo_es_hibrido"]

# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.header("Contexto a predecir")

    anio_min = min(anios_entrenamiento) if anios_entrenamiento else 2016
    anio_max_train = max(anios_disponibles) if anios_disponibles else 2025
    anio_default = anio_max_train + 1
    anio = st.number_input(
        "Año",
        min_value=anio_min,
        max_value=anio_min + 20,
        value=anio_default,
        step=1,
        help=(
            f"Años disponibles en entrenamiento: {anios_disponibles}. "
            + (
                "Para años futuros, la tendencia log-lineal por barrio extrapola "
                "el nivel anual, mientras que la forma espacio-temporal viene del HistGB."
                if modelo_es_hibrido
                else "Los árboles no extrapolan: años fuera del rango se tratan como el último visto."
            )
        ),
    )
    if not modelo_es_hibrido and anio > anio_max_train:
        st.caption(f"Año fuera del rango de entrenamiento ({anios_disponibles}).")

    mes_num = st.selectbox(
        "Mes", list(range(1, 13)), index=5, format_func=lambda m: MESES[m - 1]
    )
    dia_semana = st.selectbox(
        "Día de la semana", list(range(7)), index=4, format_func=lambda i: DIAS[i]
    )
    turno = st.selectbox("Turno", TURNOS, index=3)

    feriados = feriados_anio(int(anio))
    es_feriado_default = (mes_num, 1) in feriados
    es_feriado = st.checkbox("Es feriado", value=bool(es_feriado_default))

    st.markdown("---")
    metrica = st.radio(
        "Métrica del mapa",
        ["Cantidad predicha", "Riesgo (alto/bajo)"],
        index=0,
    )
    modo = st.radio(
        "Modo de visualización",
        ["Absoluto", "Relativo al barrio (Δ vs. promedio)"],
        index=0,
        help=(
            "Barrio es la variable más importante del modelo. En modo Absoluto, "
            "barrios grandes como Palermo siempre dominan. En modo Relativo se "
            "muestra cuánto sube/baja la predicción de este contexto vs. el "
            "promedio del barrio (entre los 4 turnos), para resaltar el efecto "
            "del turno/día/feriado."
        ),
    )
es_fin_de_semana = dia_semana >= 5

# ─────────────────────────── Predicción por barrio ───────────────────────────
primer_b_geo = list(barrio_geo_to_train.keys())[0]
primer_b_train = barrio_geo_to_train[primer_b_geo]
primer_comuna = barrio_comuna[primer_b_train]

barrio_base_payload = build_payload(primer_b_train, primer_comuna, turno, dia_semana, mes_num, anio, es_fin_de_semana, es_feriado)
barrios_lista_payload = []
barrios_validos_geo = []

for b_geo, b_train in barrio_geo_to_train.items():
    comuna = barrio_comuna[b_train]
    barrios_lista_payload.append({"barrio": b_train, "comuna": comuna})
    barrios_validos_geo.append(b_geo)

# Petición Batch única
batch_response = api_post("/predict/batch-barrio", {
    "barrio_base": barrio_base_payload,
    "barrios": barrios_lista_payload
})

# Mapeamos los resultados de vuelta alineados con el geojson
df_pred = pd.DataFrame(batch_response)
df_pred["barrio_geo"] = barrios_validos_geo
df_pred["prob_alto_riesgo"] = df_pred["probabilidad_alto_riesgo"]
df_pred["riesgo_int"] = np.where(df_pred["riesgo"] == "Alto", 1, 0)

# ─────────────────────────── UI ───────────────────────────

st.title("Mapa de calor — Predicción de delitos en CABA")
st.caption(
    f"Clasificación: riesgo umbral {umbral} · "
    f"Años entrenamiento: {anios_entrenamiento}"
)
if modelo_es_hibrido:
    barrio_top_factor = df_pred.loc[df_pred["factor_anio"].idxmax()]
    max_conocido = max(anios_disponibles) if anios_disponibles else max(anios_entrenamiento)
    regimen = "interpolación" if int(anio) <= max_conocido else "extrapolación"
    st.caption(
        f"Año {anio} ({regimen} respecto del rango conocido ≤ {max_conocido}) · "
        f"factor por barrio: min {df_pred['factor_anio'].min():.2f}, "
        f"max {df_pred['factor_anio'].max():.2f} ({barrio_top_factor['barrio']})"
    )

col_map, col_side = st.columns([2, 1])

if metrica.startswith("Cantidad"):
    if modo.startswith("Absoluto"):
        valor_col = "cantidad_predicha"
        legend = "Cantidad de delitos predicha (acumulado anual del bucket)"
        fill_color = "YlOrRd"
    else:
        valor_col = "delta_vs_baseline"
        legend = "Δ vs. promedio del barrio (positivo = más crimen que lo típico)"
        fill_color = "RdYlGn_r"
else:
    valor_col = "prob_alto_riesgo"
    legend = "Probabilidad de alto riesgo"
    fill_color = "OrRd"

with col_map:
    m = folium.Map(location=[-34.61, -58.45], zoom_start=12, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=geojson,
        data=df_pred,
        columns=["barrio_geo", valor_col],
        key_on=f"feature.properties.{clave_barrio_geo}",
        fill_color=fill_color,
        fill_opacity=0.78,
        line_opacity=0.3,
        nan_fill_color="lightgray",
        legend_name=legend,
    ).add_to(m)

    lookup = {row["barrio_geo"]: row for _, row in df_pred.iterrows()}
    for f in geojson["features"]:
        zona = f["properties"][clave_barrio_geo]
        row = lookup.get(zona)
        if row is None:
            continue
        tooltip_html = (
            f"<b>{zona}</b> (Comuna {row['comuna']})<br>"
            f"Cantidad predicha: <b>{row['cantidad_predicha']:.2f}</b><br>"
            f"Promedio del barrio: {row['cantidad_baseline']:.2f}<br>"
            f"Δ vs. promedio: <b>{row['delta_vs_baseline']:+.2f}</b><br>"
            f"Factor del año: {row['factor_anio']:.2f}<br>"
            f"Riesgo: <b>{row['riesgo']}</b> "
            f"(P(alto)={row['prob_alto_riesgo']:.2f})"
        )
        folium.GeoJson(
            f,
            style_function=lambda *_: {"fillOpacity": 0, "color": "transparent"},
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
        ).add_to(m)

    st_folium(m, height=620, use_container_width=True, returned_objects=[])

with col_side:
    st.subheader("Top 10 barrios")
    top = (
        df_pred.sort_values("cantidad_predicha", ascending=False)
        .head(10)[["barrio_geo", "comuna", "cantidad_predicha", "riesgo"]]
        .rename(columns={"barrio_geo": "Barrio", "comuna": "Comuna",
                         "cantidad_predicha": "Cantidad", "riesgo": "Riesgo"})
        .reset_index(drop=True)
    )
    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Resumen")
    st.metric("Barrios cubiertos", f"{len(df_pred)} / {len(barrios_geojson)}")
    st.metric("Cantidad predicha total", f"{df_pred['cantidad_predicha'].sum():.0f}")
    st.metric("Barrios de alto riesgo", int(df_pred["riesgo_int"].sum()))

st.markdown("---")

# ─────────────────────────── Detalle por barrio ───────────────────────────
st.subheader("Predicción por turno — para un barrio específico")
barrio_focus = st.selectbox(
    "Barrio",
    sorted(df_pred["barrio_geo"]),
    index=0,
)
b_train = barrio_geo_to_train[barrio_focus]
comuna_focus = barrio_comuna[b_train]
factor_focus = api_post("/factor-anio", {
    "barrio": b_train,
    "anio": int(anio)
})

payload_turno_focus = build_payload(b_train, comuna_focus, turno, dia_semana, mes_num, anio, es_fin_de_semana, es_feriado)
turnos_response = api_post("/predict/turnos", payload_turno_focus)
df_turno = pd.DataFrame(turnos_response)

c1, c2 = st.columns(2)
with c1:
    st.caption(
        f"Cantidad predicha por turno — {barrio_focus} (Comuna {comuna_focus}) "
        f"· factor año {anio}: {factor_focus:.2f}"
    )
    chart_cant = (
        alt.Chart(df_turno)
        .mark_bar()
        .encode(
            x=alt.X("turno:N", sort=TURNOS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("cantidad_predicha:Q", title=None),
        )
    )
    st.altair_chart(chart_cant, use_container_width=True)
with c2:
    st.caption(f"Probabilidad de alto riesgo por turno — {barrio_focus}")
    chart_prob = (
        alt.Chart(df_turno)
        .mark_bar()
        .encode(
            x=alt.X("turno:N", sort=TURNOS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("probabilidad_alto_riesgo:Q", title=None),
        )
    )
    st.altair_chart(chart_prob, use_container_width=True)

# ─────────────────────────── Tendencia temporal ───────────────────────────
if modelo_es_hibrido:
    st.markdown("---")
    st.subheader(f"Evolución temporal estimada — {barrio_focus}")
    anios_grafico = list(range(min(anios_disponibles), int(anio) + 6))

    payload_evolucion = build_payload(b_train, comuna_focus, turno, dia_semana, mes_num, anio, es_fin_de_semana, es_feriado)
    payload_evolucion["anios"] = list(anios_grafico)

    evolucion_response = api_post("/predict/evolucion-temporal", payload_evolucion)
    serie = pd.DataFrame(evolucion_response["serie"])
    serie.index = serie.index.astype(str)

    chart = (
        alt.Chart(serie)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "anio:O",
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "cantidad_estimada:Q",
                title=None,
            ),
        )
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Cantidad predicha para este contexto ({DIAS[dia_semana]}, {MESES[mes_num - 1]}, {turno}) "
        f"a lo largo de los años, aplicando el factor de tendencia del barrio."
    )

with st.expander("Ver tabla completa de predicciones por barrio"):
    st.dataframe(
        df_pred[["barrio_geo", "comuna", "factor_anio", "cantidad_predicha", "riesgo", "prob_alto_riesgo"]]
        .rename(columns={"barrio_geo": "Barrio"})
        .sort_values("cantidad_predicha", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
