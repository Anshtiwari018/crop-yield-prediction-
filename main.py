import pandas as pd
import joblib
import requests
import numpy as np

# ================= CROP DATABASE =================
# All major crops with agronomic parameters
# Names match exactly what is in crop_production.csv
CROPS = {
    "Rice": {
        "temp": (20, 35),
        "rain": (150, 300),
        "humidity": (60, 100),
        "months": [6, 7, 8, 9],
        "season": "Kharif",
        "price": 21.83,
        "emoji": "🌾",
        "tips": [
            "Maintain 5–10 cm standing water during early vegetative stage.",
            "Apply nitrogen in 3 splits — basal, tillering, and panicle initiation.",
            "Use SRI method to improve yield by 20–30% with less water.",
        ],
    },
    "Wheat": {
        "temp": (10, 25),
        "rain": (50, 100),
        "humidity": (40, 65),
        "months": [10, 11, 12, 1],
        "season": "Rabi",
        "price": 22.75,
        "emoji": "🌿",
        "tips": [
            "Sow between Nov 1–15 in North India for optimum yield.",
            "Irrigate at crown root initiation, tillering, jointing & grain fill stages.",
            "Use disease-resistant varieties like HD-2967 or PBW-343.",
        ],
    },
    "Maize": {
        "temp": (21, 32),
        "rain": (60, 150),
        "humidity": (40, 75),
        "months": [6, 7, 8],
        "season": "Kharif",
        "price": 18.35,
        "emoji": "🌽",
        "tips": [
            "Target 60,000–65,000 plants per hectare for best canopy.",
            "Apply 120 kg N/ha in 3 equal splits at sowing, knee-high, and tasseling.",
            "Ensure proper drainage — maize cannot tolerate waterlogging for 24 hours.",
        ],
    },
    "Arhar/Tur": {
        "temp": (25, 35),
        "rain": (40, 120),
        "humidity": (30, 65),
        "months": [6, 7, 8, 9],
        "season": "Kharif",
        "price": 70.00,
        "emoji": "🫘",
        "tips": [
            "Intercrop with soybean or groundnut for better land-use efficiency.",
            "Seed inoculation with Rhizobium reduces nitrogen need by 25 kg/ha.",
            "Harvest when 75–80% of pods turn brown to minimise shattering.",
        ],
    },
    "Sugarcane": {
        "temp": (25, 35),
        "rain": (100, 250),
        "humidity": (50, 85),
        "months": [2, 3, 4, 5],
        "season": "Whole Year",
        "price": 3.40,
        "emoji": "🪵",
        "tips": [
            "Ratoon cropping reduces input cost by 30–40% vs plant crop.",
            "Drip irrigation with fertigation improves water-use efficiency by 35%.",
            "Trash mulching after harvest suppresses weeds and retains moisture.",
        ],
    },
    "Bajra": {
        "temp": (25, 40),
        "rain": (30, 80),
        "humidity": (20, 55),
        "months": [6, 7, 8],
        "season": "Kharif",
        "price": 23.50,
        "emoji": "🌾",
        "tips": [
            "Best suited for arid & semi-arid regions with low rainfall.",
            "Apply 60 kg N/ha split into 2 doses for best grain filling.",
            "Harvest when grains are hard and panicles turn grey-brown.",
        ],
    },
    "Jowar": {
        "temp": (25, 38),
        "rain": (40, 100),
        "humidity": (25, 60),
        "months": [6, 7, 8],
        "season": "Kharif",
        "price": 28.40,
        "emoji": "🌿",
        "tips": [
            "Dual-purpose crop — grain and fodder both have high value.",
            "Apply 80 kg N/ha; split application improves grain weight.",
            "Deep black soils of Deccan give best Jowar yields.",
        ],
    },
    "Groundnut": {
        "temp": (22, 33),
        "rain": (50, 120),
        "humidity": (40, 70),
        "months": [6, 7, 8],
        "season": "Kharif",
        "price": 55.50,
        "emoji": "🥜",
        "tips": [
            "Light sandy loam soil with good drainage is ideal.",
            "Apply gypsum (400 kg/ha) at pegging stage for better pod fill.",
            "Maintain soil moisture at peg zone critical for yield.",
        ],
    },
    "Gram": {
        "temp": (10, 25),
        "rain": (40, 80),
        "humidity": (30, 55),
        "months": [10, 11, 12],
        "season": "Rabi",
        "price": 53.40,
        "emoji": "🫘",
        "tips": [
            "Needs cool dry weather; excess moisture causes wilt disease.",
            "Inoculate seed with Rhizobium for better nitrogen fixation.",
            "Avoid irrigation after flowering to prevent excessive vegetative growth.",
        ],
    },
    "Rapeseed &Mustard": {
        "temp": (10, 25),
        "rain": (40, 80),
        "humidity": (35, 60),
        "months": [10, 11, 12],
        "season": "Rabi",
        "price": 52.50,
        "emoji": "🌼",
        "tips": [
            "Sow after first week of October in North India.",
            "Apply 1 kg boron/ha to improve seed set and oil content.",
            "Monitor aphid population — spray only when 30+ aphids/plant.",
        ],
    },
    "Soyabean": {
        "temp": (22, 32),
        "rain": (60, 150),
        "humidity": (50, 80),
        "months": [6, 7, 8],
        "season": "Kharif",
        "price": 39.50,
        "emoji": "🫘",
        "tips": [
            "Well-drained loamy soil with pH 6–7 gives best yield.",
            "Inoculate with Bradyrhizobium for nitrogen fixation.",
            "Harvest at 92–95% pod maturity to avoid shattering losses.",
        ],
    },
    "Potato": {
        "temp": (15, 25),
        "rain": (50, 120),
        "humidity": (55, 85),
        "months": [10, 11, 12, 1],
        "season": "Rabi",
        "price": 18.00,
        "emoji": "🥔",
        "tips": [
            "Plant certified seed tubers of 30–40 g for uniform germination.",
            "Earthing up at 25–30 DAS improves tuber development.",
            "Late blight is major threat — spray mancozeb at first sign.",
        ],
    },
    "Onion": {
        "temp": (13, 28),
        "rain": (50, 120),
        "humidity": (40, 70),
        "months": [10, 11, 12, 1, 2],
        "season": "Rabi",
        "price": 15.00,
        "emoji": "🧅",
        "tips": [
            "Transplant 6–7 week old seedlings for better bulb formation.",
            "Stop irrigation 10–12 days before harvest for proper curing.",
            "Store in well-ventilated sheds to reduce post-harvest losses.",
        ],
    },
    "Cotton(lint)": {
        "temp": (25, 38),
        "rain": (60, 150),
        "humidity": (40, 70),
        "months": [5, 6, 7],
        "season": "Kharif",
        "price": 67.00,
        "emoji": "☁️",
        "tips": [
            "Bt cotton controls bollworm but needs refuge strips nearby.",
            "Apply potash (60 kg K₂O/ha) to improve fiber quality.",
            "Monitor for whitefly and pink bollworm regularly.",
        ],
    },
    "Moong(Green Gram)": {
        "temp": (25, 35),
        "rain": (40, 90),
        "humidity": (30, 65),
        "months": [6, 7, 3, 4],
        "season": "Kharif",
        "price": 85.60,
        "emoji": "🫘",
        "tips": [
            "Short-duration crop (60–70 days) — fits well in rice-wheat rotation.",
            "Rhizobium inoculation reduces fertilizer requirement.",
            "Harvest in 2–3 pickings as pods mature at different times.",
        ],
    },
    "Barley": {
        "temp": (8, 22),
        "rain": (40, 80),
        "humidity": (30, 55),
        "months": [10, 11, 12],
        "season": "Rabi",
        "price": 17.35,
        "emoji": "🌾",
        "tips": [
            "Most drought-tolerant cereal for Rabi season.",
            "Good option in saline or alkali soils where wheat fails.",
            "Apply 60 kg N/ha for feed barley; lower for malt barley.",
        ],
    },
    "Guar seed": {
        "temp": (25, 40),
        "rain": (30, 80),
        "humidity": (20, 55),
        "months": [7, 8, 9],
        "season": "Kharif",
        "price": 45.00,
        "emoji": "🌿",
        "tips": [
            "Very drought-tolerant; suitable for arid Rajasthan conditions.",
            "Guar gum has high export demand — adds premium to farm income.",
            "Apply phosphorus (40 kg P₂O₅/ha) for good root nodulation.",
        ],
    },
    "Urad": {
        "temp": (25, 35),
        "rain": (40, 100),
        "humidity": (35, 65),
        "months": [6, 7, 8],
        "season": "Kharif",
        "price": 68.00,
        "emoji": "🫘",
        "tips": [
            "Sensitive to waterlogging — avoid heavy soils.",
            "Short-duration (65–70 days) crop good for crop rotation.",
            "Rhizobium inoculation can fix 40–50 kg N/ha.",
        ],
    },
}

STATES = [
    "Rajasthan",
    "Maharashtra",
    "Punjab",
    "Uttar Pradesh",
    "Madhya Pradesh",
    "Haryana",
    "Gujarat",
    "Bihar",
    "West Bengal",
    "Andhra Pradesh",
    "Karnataka",
    "Tamil Nadu",
    "Odisha",
    "Assam",
    "Jharkhand",
    "Telangana",
    "Chhattisgarh",
    "Uttarakhand",
    "Himachal Pradesh",
    "Jammu and Kashmir",
]

MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


# ================= LOAD MODEL =================
def load_model():
    try:
        model = joblib.load("model.pkl")
        meta = joblib.load("model_meta.pkl")
        return model, meta
    except Exception:
        return None, {}


model, model_meta = load_model()


# ================= WEATHER =================
def get_weather(city, api_key):
    try:
        if not api_key:
            return None, None, None

        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={api_key}&units=metric"
        )
        data = requests.get(url, timeout=5).json()

        if "main" not in data:
            return None, None, None

        temp = float(data["main"]["temp"])
        humidity = float(data["main"]["humidity"])
        rain = float(
            data.get("rain", {}).get("1h") or data.get("rain", {}).get("3h") or 0
        )
        return temp, humidity, rain

    except Exception:
        return None, None, None


# ================= SEASON =================
def get_season(month):
    if month in [6, 7, 8, 9]:
        return 1  # Kharif
    elif month in [10, 11, 12, 1]:
        return 2  # Rabi
    else:
        return 0  # Zaid / Whole Year


def get_season_label(month):
    if month in [6, 7, 8, 9]:
        return "Kharif"
    elif month in [10, 11, 12, 1]:
        return "Rabi"
    return "Zaid"


# ================= CROP / STATE AVERAGES =================
# Use real averages from training data if available; else fall back to constants
CROP_YIELD_AVG = {
    "Rice": 1.98,
    "Wheat": 2.09,
    "Maize": 2.14,
    "Arhar/Tur": 0.80,
    "Sugarcane": 45.7,
    "Bajra": 1.18,
    "Jowar": 1.05,
    "Groundnut": 1.20,
    "Gram": 0.84,
    "Rapeseed &Mustard": 0.77,
    "Soyabean": 1.05,
    "Potato": 12.8,
    "Onion": 11.5,
    "Cotton(lint)": 1.73,
    "Moong(Green Gram)": 0.46,
    "Barley": 1.84,
    "Guar seed": 0.98,
    "Urad": 0.50,
}

STATE_YIELD_AVG = {
    "Punjab": 8.20,
    "Haryana": 8.20,
    "West Bengal": 5.33,
    "Uttar Pradesh": 5.29,
    "Gujarat": 5.73,
    "Maharashtra": 4.40,
    "Andhra Pradesh": 4.50,
    "Karnataka": 4.43,
    "Tamil Nadu": 4.20,
    "Madhya Pradesh": 3.21,
    "Rajasthan": 2.93,
    "Bihar": 3.51,
    "Odisha": 3.73,
    "Assam": 3.50,
    "Jharkhand": 2.73,
    "Telangana": 4.50,
    "Chhattisgarh": 2.18,
    "Uttarakhand": 3.32,
    "Himachal Pradesh": 1.95,
    "Jammu and Kashmir": 1.56,
}


def get_crop_avg(crop):
    # Prefer real data from model training
    if model_meta.get("crop_avg_map"):
        return model_meta["crop_avg_map"].get(crop, CROP_YIELD_AVG.get(crop, 2.0))
    return CROP_YIELD_AVG.get(crop, 2.0)


def get_state_avg(state):
    if model_meta.get("state_avg_map"):
        return model_meta["state_avg_map"].get(state, STATE_YIELD_AVG.get(state, 3.0))
    return STATE_YIELD_AVG.get(state, 3.0)


# ================= ENCODING =================
def encode_crop(crop):
    cats = model_meta.get("crop_cats", [])
    if crop in cats:
        return cats.index(crop)
    return len(cats)  # unseen → last index


def encode_state(state):
    cats = model_meta.get("state_cats", [])
    if state in cats:
        return cats.index(state)
    return len(cats)


# ================= ML PREDICTION =================
def predict_yield(area, temp, rain, humidity, crop, state, month):
    if model is None:
        return None, 40

    try:
        df = pd.DataFrame(
            [
                {
                    "area": area,
                    "log_area": np.log1p(area),
                    "temp": temp,
                    "rain": rain,
                    "humidity": humidity,
                    "crop": encode_crop(crop),
                    "state": encode_state(state),
                    "season": get_season(month),
                    "crop_avg": get_crop_avg(crop),
                    "state_avg": get_state_avg(state),
                }
            ]
        )

        pred = model.predict(df)[0]
        yield_ton = np.expm1(pred)
        yield_ton = max(0.5, min(yield_ton, area * 10))  # realistic cap

        # Confidence based on how well weather matches crop requirements
        c = CROPS.get(crop, {})
        matches = sum(
            [
                c.get("temp", (0, 100))[0] <= temp <= c.get("temp", (0, 100))[1],
                c.get("rain", (0, 500))[0] <= rain <= c.get("rain", (0, 500))[1],
                c.get("humidity", (0, 100))[0]
                <= humidity
                <= c.get("humidity", (0, 100))[1],
                month in c.get("months", []),
            ]
        )
        confidence = 60 + matches * 8  # 60–92%

        return round(yield_ton, 2), int(confidence)
    except Exception:
        return None, 40


# ================= CROP RECOMMENDATION =================
def analyze_crop(temp, rain, humidity, month):
    results = {}

    for crop, data in CROPS.items():
        score = 0

        # Temperature fit
        t_lo, t_hi = data["temp"]
        if t_lo <= temp <= t_hi:
            score += 30
        elif temp < t_lo:
            score += max(0, 30 - (t_lo - temp) * 2)
        else:
            score += max(0, 30 - (temp - t_hi) * 2)

        # Rainfall fit
        r_lo, r_hi = data["rain"]
        if r_lo <= rain <= r_hi:
            score += 25
        elif rain < r_lo:
            score += max(0, 25 - (r_lo - rain) * 0.3)
        else:
            score += max(0, 25 - (rain - r_hi) * 0.3)

        # Humidity fit
        h_lo, h_hi = data["humidity"]
        if h_lo <= humidity <= h_hi:
            score += 20

        # Season / month fit — strong signal
        if month in data["months"]:
            score += 25
        else:
            score -= 35  # penalise wrong season heavily

        results[crop] = max(0, int(score))

    best = max(results, key=results.get)
    return best, results


# ================= CROP DETAILS =================
def get_details(crop):
    c = CROPS[crop]
    return {
        "temp": f"{c['temp'][0]}–{c['temp'][1]} °C",
        "rain": f"{c['rain'][0]}–{c['rain'][1]} mm",
        "humidity": f"{c['humidity'][0]}–{c['humidity'][1]} %",
        "months": c["months"],
        "season": c["season"],
    }


# ================= PROFIT CALCULATION =================
def calculate_profit(crop, yield_kg):
    """
    Uses MSP (Minimum Support Price) from CROPS dict.
    Input cost assumed ₹8/kg (seeds, fertiliser, labour).
    """
    price = CROPS[crop]["price"]
    income = yield_kg * price
    cost = yield_kg * 8
    profit = income - cost
    return int(income), int(cost), int(profit)


# ================= MAIN RUN FUNCTION =================
def run_prediction(area, city, month, api_key, state):

    # ✅ state clean (important for encoding)
    state = state.lower().strip()

    # 🌐 get weather
    temp, humidity, rain = get_weather(city, api_key)

    # ✅ API fail handle
    if temp is None:
        temp, humidity, rain = 28, 60, 50  

    # 🌱 crop analysis
    crop, scores = analyze_crop(temp, rain, humidity, month)

    # 🤖 ML prediction
    yield_ton, confidence = predict_yield(
        area, temp, rain, humidity, crop, state, month
    )

    
    if yield_ton is None:
        return {"error": "Model failed"}

    # 📊 calculations
    yield_kg = yield_ton * 1000
    details = get_details(crop)
    income, cost, profit = calculate_profit(crop, yield_kg)

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
        "cost": cost,
        "profit": profit,
        "scores": scores,
    }
