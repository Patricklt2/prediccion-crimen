from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    barrio: str
    comuna: str
    anio: int
    mes_num: int
    dia_semana: int
    turno: str
    es_fin_de_semana: bool
    es_feriado: bool

class PredictResponse(BaseModel):
    cantidad_predicha: float
    riesgo: str
    probabilidad_alto_riesgo: float
    factor_anio: float

class BarrioItem(BaseModel):
    barrio: str
    comuna: str

class BatchBarrioRequest(BaseModel):
    barrio_base: PredictRequest
    barrios: List[BarrioItem]

class MetadataResponse(BaseModel):
    umbral: float | None
    anios_entrenamiento: List[int]
    anios_disponibles: List[int]
    modelo_es_hibrido: bool
class FactorAnioRequest(BaseModel):
    barrio: str
    anio: int

class FactorAnioResponse(BaseModel):
    factor_anio: float

class BarrioPredictionResponse(BaseModel):
    barrio: str
    comuna: str
    cantidad_predicha: float
    cantidad_baseline: float  # <--- NUEVO: Promedio de los 4 turnos
    delta_vs_baseline: float  # <--- NUEVO: Cantidad - Baseline
    riesgo: str
    probabilidad_alto_riesgo: float
    factor_anio: float

class EvolucionTemporalRequest(BaseModel):
    barrio: str
    comuna: str
    mes_num: int
    dia_semana: int
    turno: str
    es_fin_de_semana: bool
    es_feriado: bool
    anios: List[int]

class EvolucionTemporalItem(BaseModel):
    anio: int
    cantidad_estimada: float

class EvolucionTemporalResponse(BaseModel):
    serie: List[EvolucionTemporalItem]