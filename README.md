# Seguridad CABA — UI de predicciones

Aplicación Streamlit que expone los modelos del notebook
`Tercer_Entrega_Ciencia_de_Datos_v3.ipynb` (modelo híbrido):

- **HistGradientBoosting** (sin `anio`) — patrón espacio-temporal promedio.
- **Tendencia anual log-lineal por barrio** (con shrinkage + cap) — nivel anual,
  permite extrapolar a años futuros.
- **Clasificador de riesgo** — etiqueta el contexto (barrio, día, turno, etc.)
  como riesgo bajo / medio / alto.

Predicción final de cantidad: `HistGB(contexto) × factor_tendencia(barrio, año)`.

La arquitectura está separada en dos servicios: un **backend FastAPI** que sirve
los modelos, y una **UI Streamlit** que consume ese backend por HTTP. La UI no
carga los `.joblib`; toda la inferencia ocurre en la API.

## Estructura

```
.
├── backend/                                # API FastAPI (sirve los modelos)
│   ├── main.py                             # app FastAPI + rutas + CORS
│   ├── schemas.py                          # modelos Pydantic (request/response)
│   ├── model_loader.py                     # carga los .joblib desde models/
│   ├── predictor.py                        # lógica de inferencia (modelo híbrido)
│   ├── utils.py                            # factor de tendencia por barrio/año
│   ├── requirements.txt
│   └── runtime.txt                         # python-3.11.10 (deploy)
├── app/                                    # UI Streamlit
│   ├── streamlit_app.py                    # UI (Streamlit + folium)
│   ├── api.py                              # cliente HTTP hacia el backend
│   ├── data.py                             # GeoJSON de barrios, feriados, etc.
│   ├── utils.py
│   └── requirements.txt
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

## API (backend FastAPI)

El backend (`backend/main.py`) carga los modelos desde `models/` al arrancar y
expone la inferencia vía HTTP. La UI lo consume a través de `app/api.py`, que
maneja el *cold start* del free tier (reintenta con timeout extendido).

**URL principal de la API:** https://prediccion-crimen-1.onrender.com

Docs interactivas (Swagger) en
[`/docs`](https://prediccion-crimen-1.onrender.com/docs).

### Endpoints

| Método | Ruta                          | Descripción                                                               |
|--------|-------------------------------|---------------------------------------------------------------------------|
| `GET`  | `/health`                     | Health check (`{"status": "ok"}`).                                        |
| `GET`  | `/metadata`                   | Umbral del clasificador, años de entrenamiento/disponibles, flag híbrido. |
| `POST` | `/predict/turnos`             | Predice cantidad + riesgo para los 4 turnos de un contexto.               |
| `POST` | `/predict/batch-barrio`       | Predice una lista de barrios vs. un barrio base (delta vs. baseline).     |
| `POST` | `/predict/evolucion-temporal` | Serie de cantidad estimada por año (extrapola con el factor de tendencia).|
| `POST` | `/factor-anio`                | Factor de tendencia de un barrio para un año dado.                        |

Los esquemas de request/response están definidos con Pydantic en
`backend/schemas.py`. CORS está restringido al dominio de la UI en Streamlit
Cloud (ver `allow_origins` en `backend/main.py`).

### Correr la API local

```bash
pip install -r backend/requirements.txt
uvicorn main:app --reload --app-dir backend
```

## UI (Streamlit)

**URL de la UI:** https://prediccion-crimen-wh9gemhxpxjhsykgfcgwt7.streamlit.app/

### Correr la UI local

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

La primera carga descarga el GeoJSON de barrios de CABA y el calendario de
feriados del año seleccionado, así que requiere conexión a internet. La UI
también necesita el backend en línea (ver `API_URL` en `app/api.py`).

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
