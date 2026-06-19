import requests

API_BASE_URL = "http://localhost:8000"   # change this after deployment

def get_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None

def get_available_routes():
    try:
        r = requests.get(f"{API_BASE_URL}/routes", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return {"prophet_routes": [], "lstm_routes": []}

def predict_price(source_city, destination_city, airline, days_left,
                   stops="zero", travel_class="Economy",
                   departure_time="Morning", arrival_time="Evening",
                   duration_hours=2.5):
    payload = {
        "source_city": source_city,
        "destination_city": destination_city,
        "airline": airline,
        "days_left": days_left,
        "stops": stops,
        "travel_class": travel_class,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "duration_hours": duration_hours
    }
    try:
        r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

def get_history(source_city, destination_city, limit=100):
    try:
        r = requests.get(
            f"{API_BASE_URL}/history",
            params={"source_city": source_city, "destination_city": destination_city, "limit": limit},
            timeout=5
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)