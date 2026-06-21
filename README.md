# Seguridad CABA — UI de predicciones

Aplicación Streamlit que expone los modelos del notebook
`Tercer_Entrega_Ciencia_de_Datos_v3.ipynb` (modelo híbrido):

- **HistGradientBoosting** (sin `anio`) — patrón espacio-temporal promedio.
- **Tendencia anual log-lineal por barrio** (con shrinkage + cap) — nivel anual,
  permite extrapolar a años futuros.
- **Clasificador de riesgo** — etiqueta el contexto (barrio, día, turno, etc.)
  como riesgo bajo / medio / alto.

Predicción final de cantidad: `HistGB(contexto) × factor_tendencia(barrio, año)`.

## Estructura

```
.
├── app/streamlit_app.py                    # UI (Streamlit + folium)
├── models/                                 # .joblib generados por el notebook
│   ├── modelo_regresion_delitos.joblib
│   ├── modelo_clasificacion_riesgo.joblib
│   ├── tendencia_barrio.joblib             # tendencia con shrinkage + cap
│   ├── tendencia_barrio_cruda.joblib       # tendencia sin regularizar (debug)
│   └── metadata_modelos.joblib
├── delitos_2016.csv … delitos_2025.csv     # datasets anuales (CABA)
├── Tercer_Entrega_Ciencia_de_Datos_v3.ipynb
├── Tercer_Entrega_Ciencia_de_Datos_v2.ipynb
└── requirements.txt
```

> La app solo necesita `delitos_2023.csv` en runtime (para mapear
> barrio → comuna). El resto de los CSVs se usan únicamente al reentrenar.

## Correr local

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

La primera carga descarga el GeoJSON de barrios de CABA y el calendario de
feriados del año seleccionado, así que requiere conexión a internet.

## Deploy en Streamlit Community Cloud

1. Entrar a https://share.streamlit.io con la cuenta de GitHub.
2. *New app* → repo `Patricklt2/prediccion-crimen`, branch `main`,
   file `app/streamlit_app.py`.
3. *Deploy*. Streamlit instala `requirements.txt` y publica una URL pública.

## Reentrenar los modelos

Los CSVs anuales (`delitos_2016.csv`, …, `delitos_2025.csv`) ya están versionados
en la raíz del repo. Para regenerar los `.joblib` de `models/`, abrir y ejecutar
`Tercer_Entrega_Ciencia_de_Datos_v3.ipynb` de punta a punta.

Fuente de los datos: portal de datos abiertos del Gobierno de la Ciudad de
Buenos Aires (mapa del delito).
