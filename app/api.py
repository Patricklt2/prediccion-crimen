import requests

API_URL = "https://prediccion-crimen-1.onrender.com"

def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()

def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()
