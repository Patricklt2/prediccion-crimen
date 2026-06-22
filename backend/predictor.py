import numpy as np
import pandas as pd

from backend.model_loader import reg, clf, features, tendencia, meta
from backend.utils import factor_anio
from backend.schemas import MetadataResponse


TURNOS = ["Madrugada", "Mañana", "Tarde", "Noche"]

def build_row(payload: dict, turno: str) -> dict:
    return {
        "barrio": payload["barrio"],
        "comuna": payload["comuna"],
        "dia_semana": payload["dia_semana"],
        "mes_num": payload["mes_num"],
        "turno": turno,
        "es_fin_de_semana": payload["es_fin_de_semana"],
        "es_feriado": payload["es_feriado"],
    }

def predict_turnos(payload: dict):
    results = []

    for t in TURNOS:
        fila = build_row(
            {
                **payload,
                "turno": t,
            },
            t,
        )

        X = pd.DataFrame([fila])[features]

        base = float(np.clip(reg.predict(X)[0], 0, None))
        factor = factor_anio(payload["barrio"], payload["anio"], tendencia)

        cantidad = base * factor

        riesgo = int(clf.predict(X)[0])

        try:
            prob = float(clf.predict_proba(X)[0][1])
        except Exception:
            prob = float(riesgo)

        results.append({
            "turno": t,
            "cantidad_predicha": round(cantidad, 2),
            "riesgo": "Alto" if riesgo == 1 else "Bajo",
            "probabilidad_alto_riesgo": round(prob, 3),
        })

    return results

def metadata():
    return MetadataResponse(
        umbral=meta.get("umbral_clasificacion"),
        anios_entrenamiento=meta.get("anios_entrenamiento", [2022, 2023]),
        anios_disponibles=meta.get(
            "anios_disponibles",
            meta.get("anios_entrenamiento", [2022, 2023])
        ),
        modelo_es_hibrido = tendencia is not None and "anio" not in features
    )

def get_factor_anio(payload: dict):
    return factor_anio(payload["barrio"], payload["anio"], tendencia)

def predict_batch_barrio(payload: dict, barrios: list[dict]):
    filas = []
    for b in barrios:
        fila = build_row({**payload, "barrio": b["barrio"], "comuna": b["comuna"]}, payload["turno"])
        filas.append(fila)

    X = pd.DataFrame(filas)[features]
    base = np.clip(reg.predict(X), 0, None)
    factors = np.array([factor_anio(f["barrio"], payload["anio"], tendencia) for f in filas])
    cantidad = base * factors
    riesgo = clf.predict(X)
    
    try:
        prob = clf.predict_proba(X)[:, 1]
    except Exception:
        prob = riesgo.astype(float)

    # ──── CÁLCULO DEL BASELINE (4 turnos) ────
    filas_baseline = []
    for f in filas:
        for t in TURNOS:
            f2 = dict(f)
            f2["turno"] = t
            filas_baseline.append(f2)
            
    X_base = pd.DataFrame(filas_baseline)[features]
    tree_base = np.clip(reg.predict(X_base), 0, None).reshape(len(filas), len(TURNOS))
    cant_base = tree_base * factors[:, None]
    promedios_baseline = cant_base.mean(axis=1)

    return [
        {
            "barrio": filas[i]["barrio"],
            "comuna": filas[i]["comuna"],
            "cantidad_predicha": round(float(cantidad[i]), 2),
            "cantidad_baseline": round(float(promedios_baseline[i]), 2),
            "delta_vs_baseline": round(float(cantidad[i] - promedios_baseline[i]), 2),
            "riesgo": "Alto" if int(riesgo[i]) == 1 else "Bajo",
            "probabilidad_alto_riesgo": round(float(prob[i]), 3),
            "factor_anio": round(float(factors[i]), 3),
        }
        for i in range(len(filas))
    ]

def predict_evolucion_temporal(payload: dict, anios: list[int]):
    results = []
    fila_ref = build_row(payload, payload["turno"])
    X = pd.DataFrame([fila_ref])[features]
    base_ref = float(np.clip(reg.predict(X)[0], 0, None))
    
    for a in anios:
        factor = factor_anio(payload["barrio"], a, tendencia)
        results.append({
            "anio": a,
            "cantidad_estimada": round(base_ref * factor, 2)
        })
    return {"serie": results}