from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

reg = joblib.load(MODELS_DIR / "modelo_regresion_delitos.joblib")
clf = joblib.load(MODELS_DIR / "modelo_clasificacion_riesgo.joblib")
meta = joblib.load(MODELS_DIR / "metadata_modelos.joblib")

tendencia_path = MODELS_DIR / "tendencia_barrio.joblib"
tendencia = (
    joblib.load(tendencia_path)
    if tendencia_path.exists()
    else None
)

features = meta["features"]