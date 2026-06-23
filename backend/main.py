from fastapi import FastAPI
from schemas import (
    PredictRequest, 
    BatchBarrioRequest,
    FactorAnioRequest,
    EvolucionTemporalRequest
)
from predictor import (
    predict_batch_barrio,
    predict_turnos,
    metadata,
    get_factor_anio,
    predict_evolucion_temporal
)

app = FastAPI(
    title="API Seguridad Ciudadana",
    version="1.0"
)

# permitir acceso desde streamlit
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prediccion-crimen-wh9gemhxpxjhsykgfcgwt7.streamlit.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/turnos")
def predict_turnos_endpoint(payload: PredictRequest):
    return predict_turnos(payload.model_dump())

@app.get("/metadata")
def get_metadata():
    return metadata()

@app.post("/factor-anio")
def get_factoranio(payload: FactorAnioRequest):
    return get_factor_anio(payload.model_dump())

@app.post("/predict/batch-barrio") # Modificado internamente
def predict_batch(payload: BatchBarrioRequest):
    return predict_batch_barrio(
        payload.barrio_base.model_dump(),
        [b.model_dump() for b in payload.barrios],
    )

@app.post("/predict/evolucion-temporal") # <--- NUEVO ENDPOINT
def get_evolucion_temporal(payload: EvolucionTemporalRequest):
    data = payload.model_dump()
    return predict_evolucion_temporal(data, data["anios"])