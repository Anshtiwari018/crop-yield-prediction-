import pandas as pd
import joblib
import requests
import numpy as np

# ================= LOAD MODEL =================
def load_model():
    try:
        return joblib.load("model.pkl")
    except:
        return None

model = load_model()

# ================= WEATHER =================
def get_weather(city, api_key):
    try:
        if not api_key:
            return 28, 60, 50

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        data = requests.get(url, timeout=5).json()

        if "main" not in data:
            return 28, 60, 50

        return (
            data["main"]["temp"],
            data["main"]["humidity"],
            data.get("rain", {}).get("1h", 0)
        )
    except:
        return 28, 60, 50

# ================= CROP DATABASE =================
CROPS = {
    "Rice": {"temp": (20,35), "rain": (150,300), "humidity": (60,100), "months":[6,7,8,9], "price":20},
    "Wheat": {"temp": (10,25), "rain": (50,100), "humidity": (40,60), "months":[10,11,12,1], "price":25},
    "Maize": {"temp": (21,30), "rain": (50,150), "humidity": (40,70), "months":[6,7,8], "price":18},
    "Arhar": {"temp": (25,35), "rain": (40,100), "humidity": (30,60), "months":[6,7,8,9], "price":70},
    "Sugarcane": {"temp": (25,35), "rain": (100,250), "humidity": (50,80), "months":[2,3,4,5], "price":3}
}

# ================= SEASON =================
def get_season(month):
    if month in [6,7,8,9]:
        return 1
    elif month in [10,11,12,1]:
        return 2
    else:
        return 0

# ================= SMART ENCODING =================
def encode_crop(crop):
    return list(CROPS.keys()).index(crop) + 1

def encode_state(state):
    states = ["Rajasthan","Maharashtra","Punjab"]
    return states.index(state) + 1 if state in states else 0

# ================= SMART AVG =================
def estimate_crop_avg(crop):
    base = {
        "Rice": 60,
        "Wheat": 50,
        "Maize": 45,
        "Arhar": 35,
        "Sugarcane": 80
    }
    return base.get(crop, 50)

def estimate_state_avg(state):
    base = {
        "Rajasthan": 40,
        "Maharashtra": 55,
        "Punjab": 70
    }
    return base.get(state, 50)

# ================= ML =================
def predict_yield(area, temp, rain, humidity, crop, state, month):

    if model is None:
        return None, 40

    try:
        df = pd.DataFrame([{
            "area": area,
            "log_area": np.log1p(area),
            "temp": temp,
            "rain": rain,
            "humidity": humidity,
            "crop": encode_crop(crop),
            "state": encode_state(state),
            "season": get_season(month),
            "crop_avg": estimate_crop_avg(crop),
            "state_avg": estimate_state_avg(state)
        }])

        pred = model.predict(df)[0]
        yield_ton = np.expm1(pred)

        # 🔥 smoothing (realistic output)
        yield_ton = max(0.5, min(yield_ton, area * 10))

        confidence = int(np.clip(80, 65, 95))

        return round(yield_ton, 2), confidence

    except:
        return None, 40

# ================= CROP LOGIC =================
def analyze_crop(temp, rain, humidity, month):
    results = {}

    for crop, data in CROPS.items():
        score = 0

        if data["temp"][0] <= temp <= data["temp"][1]:
            score += 30
        if data["rain"][0] <= rain <= data["rain"][1]:
            score += 30
        if data["humidity"][0] <= humidity <= data["humidity"][1]:
            score += 20
        if month in data["months"]:
            score += 20

        results[crop] = score

    best = max(results, key=results.get)
    return best, results

# ================= DETAILS =================
def get_details(crop):
    c = CROPS[crop]
    return {
        "temp": f"{c['temp'][0]}–{c['temp'][1]} °C",
        "rain": f"{c['rain'][0]}–{c['rain'][1]} mm",
        "humidity": f"{c['humidity'][0]}–{c['humidity'][1]} %",
        "months": c["months"]
    }

# ================= PROFIT =================
def calculate_profit(crop, yield_kg):
    price = CROPS[crop]["price"]
    income = yield_kg * price
    cost = yield_kg * 5
    profit = income - cost
    return int(income), int(profit)

# ================= MAIN =================
def run_prediction(area, city, month, api_key, state):

    temp, humidity, rain = get_weather(city, api_key)

    crop, scores = analyze_crop(temp, rain, humidity, month)

    yield_ton, confidence = predict_yield(
        area, temp, rain, humidity, crop, state, month
    )

    if yield_ton is None:
        yield_ton = area * 2
        confidence = 50

    yield_kg = yield_ton * 1000

    details = get_details(crop)

    income, profit = calculate_profit(crop, yield_kg)

    return {
        "yield_ton": yield_ton,
        "yield_kg": int(yield_kg),
        "confidence": confidence,
        "temp": temp,
        "rain": rain,
        "humidity": humidity,
        "crop": crop,
        "details": details,
        "income": income,
        "profit": profit,
        "scores": scores
    }