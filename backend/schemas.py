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

class EvolucionTemporalRequest(BaseModel):
    barrio: str
    comuna: str
    mes_num: int
    dia_semana: int
    turno: str
    es_fin_de_semana: bool
    es_feriado: bool
    anios: List[int]
