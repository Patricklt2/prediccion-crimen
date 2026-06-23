import pandas as pd
import requests
import streamlit as st

CSV_PATH = "delitos_2023.csv"
GEOJSON_URL = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson"

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
TURNOS = ["Madrugada", "Mañana", "Tarde", "Noche"]


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