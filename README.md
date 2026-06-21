# Seguridad CABA — UI de predicciones

Aplicación Streamlit que expone los modelos del notebook
`Tercer_Entrega_Ciencia_de_Datos_v3.ipynb` (modelo híbrido):

- **HistGradientBoosting** (sin `anio`) — patrón espacio-temporal promedio.
- **Tendencia anual log-lineal por barrio** (con shrinkage + cap) — nivel anual,
  permite extrapolar a años futuros.

Predicción final: `HistGB(contexto) × factor_tendencia(barrio, año)`.

## Estructura

```
.
├── app/streamlit_app.py            # UI (Streamlit + folium)
├── models/                         # .joblib generados por el notebook
│   ├── modelo_regresion_delitos.joblib
│   ├── modelo_clasificacion_riesgo.joblib
│   ├── tendencia_barrio.joblib
│   └── metadata_modelos.joblib
├── delitos_2023.csv                # solo para mapeo barrio→comuna
├── Tercer_Entrega_Ciencia_de_Datos_v3.ipynb
└── requirements.txt
```

## Correr local

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Deploy en Streamlit Community Cloud

1. Crear repo en GitHub y pushear este proyecto (ver pasos abajo).
2. Entrar a https://share.streamlit.io con la cuenta de GitHub.
3. *New app* → seleccionar repo, branch `main`, file `app/streamlit_app.py`.
4. *Deploy*. Streamlit instala `requirements.txt` y publica una URL pública.

### Subir a GitHub (primera vez)

```bash
# desde la carpeta del proyecto:
git init
git add .
git commit -m "Modelo v3 + Streamlit UI"

# crear el repo vacío en https://github.com/new (no marcar README ni .gitignore)
# y conectar:
git branch -M main
git remote add origin https://github.com/<TU_USUARIO>/seguridad-caba.git
git push -u origin main
```

## Reentrenar los modelos

Para reejecutar el notebook v3 hace falta tener los CSVs anuales
(`delitos_2016.csv`, …, `delitos_2025.csv`) en la raíz del proyecto. Por tamaño,
no se incluyen en el repo — bajalos del portal de datos abiertos de CABA o
desde tus enlaces de Drive.
