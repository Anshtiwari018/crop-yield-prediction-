from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv

# ================= LOAD ENV =================
load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")  # ✅ FIXED

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= LOAD DATA =================
def load_data():
    try:
        rice  = pd.read_excel(os.path.join(BASE_DIR, "Data/APY_Rice.xls"))
        wheat = pd.read_excel(os.path.join(BASE_DIR, "Data/APY_Wheat.xls"))
        arhar = pd.read_excel(os.path.join(BASE_DIR, "Data/APY_Arhar.xls"))
        print("✅ Data loaded")
        return rice, wheat, arhar
    except Exception as e:
        print("❌ Data Load Error:", e)
        return None, None, None

rice, wheat, arhar = load_data()

# ================= CLEAN DATA =================
def clean_df(df):
    if df is None:
        return None
    df = df.dropna()
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

rice  = clean_df(rice)
wheat = clean_df(wheat)
arhar = clean_df(arhar)

# ================= DISTRICT MAP =================
DISTRICT_TO_CITY = {
    "jaipur": "Jaipur",
    "alwar": "Alwar",
    "kota": "Kota",
    "udaipur": "Udaipur",
    "ajmer": "Ajmer",
    "jodhpur": "Jodhpur",
    "bikaner": "Bikaner",
    "bharatpur": "Bharatpur",
    "sikar": "Sikar",
    "pali": "Pali"
}

def map_to_city(district):
    return DISTRICT_TO_CITY.get(district.lower(), district.title())

# ================= WEATHER =================
def get_weather(city):
    if not API_KEY:
        return {"temperature": 25, "humidity": 50, "rainfall": 0}

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url, timeout=5)

        if res.status_code != 200:
            return {"temperature": 25, "humidity": 50, "rainfall": 0}

        data = res.json()
        return {
            "temperature": round(data["main"]["temp"], 1),
            "humidity": data["main"]["humidity"],
            "rainfall": round(data.get("rain", {}).get("1h", 0), 2)
        }

    except:
        return {"temperature": 25, "humidity": 50, "rainfall": 0}

# ================= YIELD COLUMN =================
def get_yield_column(df):
    for col in df.columns:
        if "yield" in col.lower():
            return col
    return None

# ================= CROP RECOMMEND =================
def recommend_crop(temp, rain, humidity, month):
    scores = {}

    scores["Rice"] = (
        (30 if 20 <= temp <= 35 else 0) +
        (35 if rain > 150 else 0) +
        (20 if humidity > 60 else 0) +
        (15 if month in [6,7,8] else 0)
    )

    scores["Wheat"] = (
        (35 if 10 <= temp <= 25 else 0) +
        (30 if 50 <= rain <= 100 else 0) +
        (20 if humidity < 70 else 0) +
        (15 if month in [10,11,12] else 0)
    )

    scores["Arhar"] = (
        (35 if 25 <= temp <= 35 else 0) +
        (30 if 60 <= rain <= 150 else 0) +
        (20 if humidity > 50 else 0)
    )

    scores["Maize"] = (
        (30 if 21 <= temp <= 30 else 0) +
        (30 if 50 <= rain <= 150 else 0) +
        (20 if humidity > 50 else 0)
    )

    scores["Sugarcane"] = (
        (35 if 25 <= temp <= 35 else 0) +
        (30 if rain > 100 else 0) +
        (20 if humidity > 70 else 0)
    )

    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "primary": sorted_crops[0][0],
        "confidence": sorted_crops[0][1],
        "alternatives": [{"crop": c, "score": s} for c, s in sorted_crops[1:3]]
    }

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        crop = data.get("crop", "").lower()
        district = data.get("district", "")
        area = float(data.get("area", 0))
        month = int(data.get("month", 6))

        if not crop or not district or area <= 0:
            return jsonify({"success": False, "error": "Invalid input"})

        city = map_to_city(district)
        weather = get_weather(city)

        crop_map = {"rice": rice, "wheat": wheat, "arhar": arhar}
        if crop not in crop_map:
            return jsonify({"success": False, "error": "Invalid crop"})

        df = crop_map[crop]
        yield_col = get_yield_column(df)

        if not yield_col:
            return jsonify({"success": False, "error": "Yield column missing"})

        if "District" in df.columns:
            df_dist = df[df["District"].str.lower() == district.lower()]
            avg_yield = df_dist[yield_col].mean() if len(df_dist) else df[yield_col].mean()
        else:
            avg_yield = df[yield_col].mean()

        prediction = round(avg_yield * area, 2)

        rec = recommend_crop(
            weather["temperature"],
            weather["rainfall"],
            weather["humidity"],
            month
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "weather": weather,
            "recommendation": rec["primary"],
            "confidence": rec["confidence"],
            "alternatives": rec["alternatives"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ================= TRENDS =================
@app.route("/analyze/trends", methods=["GET"])
def trends():
    try:
        crop = request.args.get("crop", "rice").lower()
        crop_map = {"rice": rice, "wheat": wheat, "arhar": arhar}

        if crop not in crop_map:
            return jsonify({"success": False, "error": "Invalid crop"})

        df = crop_map[crop]

        if "Year" not in df.columns:
            return jsonify({"success": False, "error": "Year column missing"})

        trend = df.groupby("Year").mean(numeric_only=True).reset_index()

        return jsonify({
            "success": True,
            "trends": trend.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)