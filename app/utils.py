import unicodedata

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()

def build_payload(barrio, comuna, turno, dia_semana, mes_num, anio, es_fin_de_semana, es_feriado):
    return {
        "barrio": barrio,
        "comuna": comuna,
        "dia_semana": int(dia_semana),
        "mes_num": int(mes_num),
        "turno": turno,
        "es_fin_de_semana": bool(es_fin_de_semana),
        "es_feriado": bool(es_feriado),
        "anio": int(anio),
    }