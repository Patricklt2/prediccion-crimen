import unicodedata
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Seguridad CABA", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CSV_PATH = ROOT / "delitos_2023.csv"
GEOJSON_URL = (
    "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/"
    "ministerio-de-educacion/barrios/barrios.geojson"
)

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
TURNOS = ["Madrugada", "Mañana", "Tarde", "Noche"]


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


@st.cache_resource
def cargar_modelos():
    reg = joblib.load(MODELS_DIR / "modelo_regresion_delitos.joblib")
    clf = joblib.load(MODELS_DIR / "modelo_clasificacion_riesgo.joblib")
    meta = joblib.load(MODELS_DIR / "metadata_modelos.joblib")
    tendencia_path = MODELS_DIR / "tendencia_barrio.joblib"
    tendencia = joblib.load(tendencia_path) if tendencia_path.exists() else None
    return reg, clf, meta, tendencia


@st.cache_data
def cargar_barrio_comuna() -> dict[str, str]:
    df = pd.read_csv(CSV_PATH, dtype={"barrio": str, "comuna": str})
    df = df[df["barrio"].notna() & df["comuna"].notna()].copy()
    df["barrio"] = df["barrio"].str.strip().str.title()
    df["comuna"] = df["comuna"].str.strip()
    counts = df.groupby(["barrio", "comuna"]).size().reset_index(name="n")
    counts = counts.sort_values(["barrio", "n"], ascending=[True, False])
    counts = counts.drop_duplicates("barrio", keep="first")
    return dict(zip(counts["barrio"], counts["comuna"]))


@st.cache_data(ttl=24 * 3600)
def cargar_geojson() -> dict:
    r = requests.get(GEOJSON_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def detectar_clave(geojson: dict, candidatos: tuple[str, ...]) -> str | None:
    props = geojson["features"][0]["properties"]
    for c in candidatos:
        if c in props:
            return c
    return None


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


reg, clf, meta, tendencia = cargar_modelos()
features: list[str] = meta["features"]
umbral = meta.get("umbral_clasificacion")
anios_entrenamiento = meta.get("anios_entrenamiento", [2022, 2023])
anios_disponibles = meta.get("anios_disponibles", anios_entrenamiento)
modelo_es_hibrido = tendencia is not None and "anio" not in features

barrio_comuna = cargar_barrio_comuna()
geojson = cargar_geojson()
clave_barrio_geo = detectar_clave(geojson, ("BARRIO", "barrio", "NOMBRE", "nombre", "Nombre"))

barrios_geojson = [f["properties"][clave_barrio_geo] for f in geojson["features"]]
train_keys = {_normalize(b): b for b in barrio_comuna}
barrio_geo_to_train: dict[str, str] = {}
for b in barrios_geojson:
    k = _normalize(b)
    if k in train_keys:
        barrio_geo_to_train[b] = train_keys[k]


def factor_anio(barrio: str, anio: int, source: dict | None = None) -> float:
    src = source if source is not None else tendencia
    if src is None:
        return 1.0
    t = src.get(barrio)
    if t is None or t.get("baseline", 0) == 0:
        return 1.0
    return float(np.exp(t["a"] + t["b"] * anio) / t["baseline"])


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


def _fila(barrio: str, comuna: str, turno_: str) -> dict:
    fila = {
        "barrio": barrio,
        "comuna": comuna,
        "dia_semana": int(dia_semana),
        "mes_num": int(mes_num),
        "turno": turno_,
        "es_fin_de_semana": bool(es_fin_de_semana),
        "es_feriado": bool(es_feriado),
    }
    if "anio" in features:  # compatibilidad con v2
        fila["anio"] = int(anio)
    return fila


# ─────────────────────────── Predicción por barrio ───────────────────────────
filas = []
barrios_validos = []
for b_geo, b_train in barrio_geo_to_train.items():
    comuna = barrio_comuna[b_train]
    filas.append(_fila(b_train, comuna, turno))
    barrios_validos.append(b_geo)

X_pred = pd.DataFrame(filas)[features]
tree_pred = np.clip(reg.predict(X_pred), 0, None)
factors = np.array([factor_anio(f["barrio"], int(anio)) for f in filas])
cantidad_pred = tree_pred * factors

riesgo_pred = clf.predict(X_pred).astype(int)
try:
    prob_alto = clf.predict_proba(X_pred)[:, 1]
except Exception:
    prob_alto = riesgo_pred.astype(float)

df_pred = pd.DataFrame({
    "barrio_geo": barrios_validos,
    "barrio": [f["barrio"] for f in filas],
    "comuna": [f["comuna"] for f in filas],
    "factor_anio": np.round(factors, 3),
    "cantidad_predicha": np.round(cantidad_pred, 2),
    "riesgo_int": riesgo_pred,
    "prob_alto_riesgo": np.round(prob_alto, 3),
})
df_pred["riesgo"] = np.where(df_pred["riesgo_int"] == 1, "Alto", "Bajo")

# Baseline por barrio: promedio sobre los 4 turnos, mismo año (el factor se cancela
# en el delta, pero queremos absolutos comparables para el tooltip).
filas_baseline = []
for f in filas:
    for t in TURNOS:
        f2 = dict(f)
        f2["turno"] = t
        filas_baseline.append(f2)
X_base = pd.DataFrame(filas_baseline)[features]
tree_base = np.clip(reg.predict(X_base), 0, None).reshape(len(filas), len(TURNOS))
# Factor por barrio constante a través de los 4 turnos del mismo año
cant_base = tree_base * factors[:, None]
df_pred["cantidad_baseline"] = np.round(cant_base.mean(axis=1), 2)
df_pred["delta_vs_baseline"] = np.round(
    df_pred["cantidad_predicha"] - df_pred["cantidad_baseline"], 2
)

# ─────────────────────────── UI ───────────────────────────
st.title("Mapa de calor — Predicción de delitos en CABA")
modelo_label = "HistGB Ajustado Anualmente" if modelo_es_hibrido else meta.get("modelo_regresion", "")
st.caption(
    f"Regresión: **{modelo_label}**  ·  "
    f"Clasificación: **{meta.get('modelo_clasificacion')}**  ·  "
    f"Umbral riesgo (mediana de cantidad): **{umbral}**  ·  "
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
    st.dataframe(top, use_container_width=True, hide_index=True)

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
factor_focus = factor_anio(b_train, int(anio))

filas_turno = [_fila(b_train, comuna_focus, t) for t in TURNOS]
X_turno = pd.DataFrame(filas_turno)[features]
cant_turno = np.clip(reg.predict(X_turno), 0, None) * factor_focus
prob_turno = clf.predict_proba(X_turno)[:, 1] if hasattr(clf, "predict_proba") else np.zeros(len(TURNOS))

df_turno = pd.DataFrame({
    "turno": TURNOS,
    "cantidad_predicha": np.round(cant_turno, 2),
    "prob_alto_riesgo": np.round(prob_turno, 3),
})

c1, c2 = st.columns(2)
with c1:
    st.caption(
        f"Cantidad predicha por turno — {barrio_focus} (Comuna {comuna_focus}) "
        f"· factor año {anio}: {factor_focus:.2f}"
    )
    st.bar_chart(df_turno.set_index("turno")["cantidad_predicha"])
with c2:
    st.caption(f"Probabilidad de alto riesgo por turno — {barrio_focus}")
    st.bar_chart(df_turno.set_index("turno")["prob_alto_riesgo"])

# ─────────────────────────── Tendencia temporal ───────────────────────────
if modelo_es_hibrido:
    st.markdown("---")
    st.subheader(f"Evolución temporal estimada — {barrio_focus}")
    anios_grafico = list(range(min(anios_disponibles), int(anio) + 6))
    fila_ref = _fila(b_train, comuna_focus, turno)
    base_ref = float(np.clip(reg.predict(pd.DataFrame([fila_ref])[features]), 0, None)[0])

    serie = pd.DataFrame({
        "anio": anios_grafico,
        "cantidad_estimada": [round(base_ref * factor_anio(b_train, a), 2)
                              for a in anios_grafico],
    }).set_index("anio")
    st.line_chart(serie)
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
