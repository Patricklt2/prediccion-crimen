import requests

API_URL = "https://prediccion-crimen-1.onrender.com"

# properly await free tier api
_COLD_START_TIMEOUT = 90
_WARM_TIMEOUT = 15

def _request(method: str, path: str, **kwargs):
    url = f"{API_URL}{path}"
    try:
        r = requests.request(method, url, timeout=_WARM_TIMEOUT, **kwargs)
    except (requests.ConnectionError, requests.Timeout):
        r = requests.request(method, url, timeout=_COLD_START_TIMEOUT, **kwargs)
    r.raise_for_status()
    return r.json()

def api_get(path: str):
    return _request("GET", path)

def api_post(path: str, payload: dict):
    return _request("POST", path, json=payload)
