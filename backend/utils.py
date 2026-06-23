import numpy as np

def factor_anio(barrio: str, anio: int, tendencia: dict | None):
    if tendencia is None:
        return 1.0

    t = tendencia.get(barrio)

    if t is None or t.get("baseline", 0) == 0:
        return 1.0

    return float(
        np.exp(t["a"] + t["b"] * anio)
        / t["baseline"]
    )