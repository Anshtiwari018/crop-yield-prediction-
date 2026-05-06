import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
try:
    WEATHER_API_KEY = st.secrets["API_KEY"]
except Exception:
    WEATHER_API_KEY = ""

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="auto",
)

# ══════════════════════════════════════════════
#  CSS — FULLY RESPONSIVE (laptop + mobile)
# ══════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f5f7f2 !important; }
.main, .main > div, .block-container { background-color: #f5f7f2 !important; }
[data-testid="column"] > div > div,
[data-testid="stVerticalBlock"] > div,
[data-testid="stHorizontalBlock"] > div,
.element-container { background-color: unset !important; }
.main .block-container {
    background: #f5f7f2 !important;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
    /* Mobile: reduce side padding */
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #1a3a2a; border-right: 1px solid #243d2e; }
section[data-testid="stSidebar"] * { color: #e8f0ec !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #2d7a46 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.65rem 1rem !important; width: 100%;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #236038 !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input {
    background: #243d2e !important; border: 1px solid #3a5c47 !important;
    border-radius: 7px !important; color: #e8f0ec !important;
}
section[data-testid="stSidebar"] hr { border-color: #2d4d3a !important; }
section[data-testid="stSidebar"] label {
    color: #8dbfa0 !important; font-size: 0.78rem !important;
    font-weight: 500 !important; letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* ── Stat Cards ── */
.stat-card {
    background: #ffffff; border-radius: 12px;
    padding: 1.1rem 1.2rem 1rem;
    border: 1px solid #e4ebe6; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 0.5rem;
}
.stat-card .label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #7a9485; margin-bottom: 4px;
}
.stat-card .value { font-size: 1.55rem; font-weight: 700; color: #1a3a2a; line-height: 1.1; }
.stat-card .sub   { font-size: 0.74rem; color: #9ab3a5; margin-top: 2px; }
.stat-card.highlight { background: #1a3a2a; border-color: #1a3a2a; }
.stat-card.highlight .label { color: #6dbf8a; }
.stat-card.highlight .value { color: #ffffff; }
.stat-card.highlight .sub   { color: #7aaa8f; }

/* Mobile stat cards — responsive font */
@media (max-width: 768px) {
    .stat-card .value { font-size: 1.15rem; }
    .stat-card .label { font-size: 0.65rem; }
    .stat-card { padding: 0.8rem 0.9rem 0.7rem; }
}

/* ── Section Headers ── */
.section-hdr {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #4a7a5a; padding-bottom: 6px;
    border-bottom: 1px solid #d8e8dd; margin-bottom: 0.9rem; margin-top: 0.3rem;
}

/* ── Panel ── */
.panel {
    background: #ffffff; border-radius: 12px; border: 1px solid #e4ebe6;
    padding: 1.3rem 1.4rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    height: 100%; margin-bottom: 0.8rem;
}
@media (max-width: 768px) {
    .panel { padding: 1rem 1rem; }
}

/* ── Progress Bars ── */
.prog-wrap { margin-bottom: 10px; }
.prog-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #5a7a6a; margin-bottom: 3px;
    flex-wrap: wrap; gap: 2px;
}
.prog-track { background: #e8f0ec; border-radius: 6px; height: 7px; overflow: hidden; }
.prog-fill  { height: 7px; border-radius: 6px; }

/* ── Tips ── */
.tip-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 0; border-bottom: 1px solid #f0f5f2;
    font-size: 0.83rem; color: #2d4d3a; line-height: 1.5;
}
.tip-row:last-child { border-bottom: none; }
.tip-dot { width:6px; height:6px; border-radius:50%; background:#2d7a46; margin-top:7px; flex-shrink:0; }

/* ── Requirements Row ── */
.req-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f0f5f2; font-size: 0.84rem;
    flex-wrap: wrap; gap: 2px;
}
.req-row:last-child { border-bottom: none; }
.req-key { color: #7a9485; font-weight: 500; }
.req-val { color: #1a3a2a; font-weight: 600; }

/* ── Crop Banner ── */
.crop-banner {
    background: linear-gradient(120deg, #1a3a2a 0%, #2d5a3d 100%);
    border-radius: 14px; padding: 1.4rem 1.8rem;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.4rem; flex-wrap: wrap; gap: 1rem;
}
.crop-banner .bleft h2 { color:#fff; font-size:1.4rem; font-weight:700; margin:0 0 4px; }
.crop-banner .bleft p  { color:#7acc96; font-size:0.82rem; margin:0; }
.crop-banner .bright   { font-size:3.2rem; line-height:1; }
.conf-pill {
    display:inline-block; background:rgba(255,255,255,0.12);
    color:#a8f0c0; border-radius:20px; padding:2px 12px;
    font-size:0.76rem; font-weight:600; margin-top:6px;
}
@media (max-width: 768px) {
    .crop-banner { padding: 1rem 1.2rem; }
    .crop-banner .bleft h2 { font-size: 1.1rem; }
    .crop-banner .bright { font-size: 2.2rem; }
}

/* ── Welcome Screen ── */
.welcome-title { font-size:2rem; font-weight:700; color:#1a3a2a; margin-bottom:4px; }
.welcome-sub   { color:#6a9478; font-size:0.94rem; margin-bottom:2rem; }
@media (max-width: 768px) {
    .welcome-title { font-size: 1.4rem; }
    .welcome-sub   { font-size: 0.82rem; }
}

.how-card {
    background:#fff; border-radius:12px; border:1px solid #e4ebe6;
    padding:1.2rem; text-align:center; margin-bottom: 0.5rem;
}
.how-card .snum {
    width:32px; height:32px; border-radius:50%; background:#e8f5ee;
    color:#2d7a46; font-weight:700; font-size:0.9rem;
    display:flex; align-items:center; justify-content:center; margin:0 auto 10px;
}
.how-card p { font-size:0.82rem; color:#5a7a6a; margin:4px 0 0; }
.how-card b { color:#1a3a2a; font-size:0.88rem; }

/* ── Crop Mini Cards ── */
.crop-mini {
    background:#fff; border-radius:10px; border:1px solid #e4ebe6;
    padding:0.9rem 0.6rem; text-align:center; margin-bottom: 0.4rem;
}
.crop-mini .ico   { font-size:1.8rem; }
.crop-mini .cname { font-size:0.82rem; font-weight:600; color:#1a3a2a; margin:4px 0 2px; }
.crop-mini .msp   { font-size:0.72rem; color:#7a9485; }

/* ── Footer ── */
.footer-bar {
    margin-top:2.5rem; padding-top:1rem; border-top:1px solid #d8e8dd;
    font-size:0.74rem; color:#9ab3a5;
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:4px;
}
@media (max-width: 768px) {
    .footer-bar { font-size: 0.65rem; flex-direction: column; gap: 6px; }
}

/* ── Streamlit overrides ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stMetricValue"] { color: #1a3a2a !important; opacity: 1 !important; }
[data-testid="stMetricLabel"] { color: #4a7a5a !important; opacity: 1 !important; }

/* ── Make dataframe scroll on mobile ── */
[data-testid="stDataFrame"] {
    overflow-x: auto !important;
}

/* ── Matplotlib charts — no overflow ── */
[data-testid="stImage"] img,
canvas {
    max-width: 100% !important;
    height: auto !important;
}

/* ── Column gaps on mobile ── */
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="column"] {
        min-width: 45% !important;
        flex: 1 1 45% !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════
#  MODEL + DATA LOAD
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        m = joblib.load("model.pkl")
        meta = joblib.load("model_meta.pkl")
        return m, meta
    except Exception:
        return None, {}


model, model_meta = load_model()

# ══════════════════════════════════════════════
#  CROP DATABASE  (18 crops from real data)
# ══════════════════════════════════════════════
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
            "Irrigate at crown root initiation, tillering, jointing & grain fill.",
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
            "Ensure drainage — maize cannot tolerate waterlogging for 24 hours.",
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
            "Ratoon cropping reduces input cost by 30–40%.",
            "Drip irrigation + fertigation improves water-use efficiency by 35%.",
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
            "Apply 80 kg N/ha; split improves grain weight significantly.",
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
            "Maintain soil moisture at peg zone — critical for yield.",
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
            "Bt cotton controls bollworm but needs refuge strips.",
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
            "Short-duration crop (60–70 days) fits well in rice-wheat rotation.",
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
            "Good option in saline/alkali soils where wheat fails.",
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
            "Very drought-tolerant; ideal for arid Rajasthan conditions.",
            "Guar gum has high export demand — adds farm income premium.",
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
            "Sensitive to waterlogging — avoid heavy clay soils.",
            "Short-duration (65–70 days) crop, good for crop rotation.",
            "Rhizobium inoculation can fix 40–50 kg N/ha.",
        ],
    },
}

STATES = [
    "Rajasthan", "Maharashtra", "Punjab", "Uttar Pradesh", "Madhya Pradesh",
    "Haryana", "Gujarat", "Bihar", "West Bengal", "Andhra Pradesh",
    "Karnataka", "Tamil Nadu", "Odisha", "Assam", "Jharkhand",
    "Telangana", "Chhattisgarh", "Uttarakhand", "Himachal Pradesh", "Jammu and Kashmir",
]

MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

CROP_YIELD_AVG = {
    "Rice": 1.98, "Wheat": 2.09, "Maize": 2.14, "Arhar/Tur": 0.80,
    "Sugarcane": 45.7, "Bajra": 1.18, "Jowar": 1.05, "Groundnut": 1.20,
    "Gram": 0.84, "Rapeseed &Mustard": 0.77, "Soyabean": 1.05,
    "Potato": 12.8, "Onion": 11.5, "Cotton(lint)": 1.73,
    "Moong(Green Gram)": 0.46, "Barley": 1.84, "Guar seed": 0.98, "Urad": 0.50,
}

STATE_YIELD_AVG = {
    "Punjab": 8.20, "Haryana": 8.20, "West Bengal": 5.33, "Uttar Pradesh": 5.29,
    "Gujarat": 5.73, "Maharashtra": 4.40, "Andhra Pradesh": 4.50, "Karnataka": 4.43,
    "Tamil Nadu": 4.20, "Madhya Pradesh": 3.21, "Rajasthan": 2.93, "Bihar": 3.51,
    "Odisha": 3.73, "Assam": 3.50, "Jharkhand": 2.73, "Telangana": 4.50,
    "Chhattisgarh": 2.18, "Uttarakhand": 3.32, "Himachal Pradesh": 1.95,
    "Jammu and Kashmir": 1.56,
}


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def get_season_id(m):
    if m in [6, 7, 8, 9]:   return 1
    if m in [10, 11, 12, 1]: return 2
    return 0

def get_season_label(m):
    if m in [6, 7, 8, 9]:   return "Kharif"
    if m in [10, 11, 12, 1]: return "Rabi"
    return "Zaid"

def get_crop_avg(crop):
    if model_meta.get("crop_avg_map"):
        return model_meta["crop_avg_map"].get(crop, CROP_YIELD_AVG.get(crop, 2.0))
    return CROP_YIELD_AVG.get(crop, 2.0)

def get_state_avg_val(state):
    if model_meta.get("state_avg_map"):
        return model_meta["state_avg_map"].get(state, STATE_YIELD_AVG.get(state, 3.0))
    return STATE_YIELD_AVG.get(state, 3.0)

def encode_crop(crop):
    cats = model_meta.get("crop_cats", [])
    return cats.index(crop) if crop in cats else len(cats)

def encode_state(state):
    cats = model_meta.get("state_cats", [])
    return cats.index(state) if state in cats else len(cats)

def get_weather(city):
    if not WEATHER_API_KEY:
        return None, None, None, False
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        d = requests.get(url, timeout=6).json()
        if d.get("cod") != 200:
            st.error(f"Weather API: {d.get('message', 'Unknown error')}")
            return None, None, None, False
        temp = float(d["main"]["temp"])
        humidity = float(d["main"]["humidity"])
        rain = float(d.get("rain", {}).get("1h") or d.get("rain", {}).get("3h") or 0)
        return temp, humidity, rain, True
    except Exception:
        return None, None, None, False

def analyze_crop(temp, rain, humidity, month):
    res = {}
    for crop, d in CROPS.items():
        s = 0
        t_lo, t_hi = d["temp"]
        if t_lo <= temp <= t_hi:   s += 30
        elif temp < t_lo:          s += max(0, 30 - int((t_lo - temp) * 2))
        else:                      s += max(0, 30 - int((temp - t_hi) * 2))

        r_lo, r_hi = d["rain"]
        if r_lo <= rain <= r_hi:   s += 25
        elif rain < r_lo:          s += max(0, 25 - int((r_lo - rain) * 0.3))
        else:                      s += max(0, 25 - int((rain - r_hi) * 0.3))

        h_lo, h_hi = d["humidity"]
        if h_lo <= humidity <= h_hi: s += 20

        if month in d["months"]: s += 25
        else:                     s -= 35

        res[crop] = max(0, s)

    return max(res, key=res.get), res

def predict_yield(area, temp, rain, humidity, crop, state, month):
    if model is None:
        return None, 40
    try:
        df = pd.DataFrame([{
            "area": area, "log_area": np.log1p(area),
            "temp": temp, "rain": rain, "humidity": humidity,
            "crop": encode_crop(crop), "state": encode_state(state),
            "season": get_season_id(month),
            "crop_avg": get_crop_avg(crop),
            "state_avg": get_state_avg_val(state),
        }])
        yt = np.expm1(model.predict(df)[0])
        yt = max(0.5, min(yt, area * 10))

        c = CROPS.get(crop, {})
        matches = sum([
            c.get("temp", (0, 100))[0] <= temp <= c.get("temp", (0, 100))[1],
            c.get("rain", (0, 500))[0] <= rain <= c.get("rain", (0, 500))[1],
            c.get("humidity", (0, 100))[0] <= humidity <= c.get("humidity", (0, 100))[1],
            month in c.get("months", []),
        ])
        confidence = 60 + matches * 8
        return round(yt, 2), int(confidence)
    except Exception:
        return None, 40

def calc_profit(crop, yield_kg):
    price = CROPS[crop]["price"]
    income = yield_kg * price
    cost = yield_kg * 8
    profit = income - cost
    return int(income), int(cost), int(profit)


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-size:1.1rem;font-weight:700;color:#e8f0ec;'>Crop Yield Prediction</div>
        <div style='font-size:0.7rem;color:#5a8a6a;margin-top:2px;
                    text-transform:uppercase;letter-spacing:0.07em;'>
            Prediction System
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

    st.markdown("**State**")
    state = st.selectbox("State", STATES, index=0, label_visibility="collapsed")

    st.markdown("**City / District**")
    city = st.text_input(
        "City", value="Jaipur", label_visibility="collapsed",
        placeholder="e.g. Jaipur, Pune, Ludhiana",
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("**Sowing Month**")
    month_num = st.selectbox(
        "Month", list(MONTHS.keys()),
        format_func=lambda x: MONTHS[x],
        index=datetime.now().month - 1,
        label_visibility="collapsed",
    )

    st.markdown("**Farm Area (hectares)**")
    area = st.number_input(
        "Area", min_value=0.1, max_value=10000.0,
        value=5.0, step=0.5, label_visibility="collapsed",
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    use_manual = st.checkbox("Override weather manually", value=False)
    if use_manual:
        st.markdown("**Temperature (°C)**")
        manual_temp = st.slider("Temp", 5, 50, 28, label_visibility="collapsed")
        st.markdown("**Humidity (%)**")
        manual_humidity = st.slider("Hum", 10, 100, 60, label_visibility="collapsed")
        st.markdown("**Rainfall (mm)**")
        manual_rain = st.slider("Rain", 0, 350, 80, label_visibility="collapsed")

    st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", use_container_width=True)

    st.markdown(
        """
    <div style='font-size:0.68rem;color:#3a5c47;margin-top:0.8rem;line-height:1.65;'>
        Weather: OpenWeatherMap API<br>
        Model: GradientBoosting Regressor<br>
        Training data: 246,091+ crop records<br>
        Crops supported: 18
    </div>
    """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
#  WELCOME SCREEN
# ══════════════════════════════════════════════
if not predict_btn:
    st.markdown(
        """
    <div class='welcome-title'>Crop Yield Prediction System</div>
    <div class='welcome-sub'>
        Enter your farm details in the sidebar to get an ML-based yield estimate,
        income forecast, and crop suitability report for 18 major Indian crops.
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, (n, title, desc) in zip(
        [c1, c2, c3],
        [
            ("1", "Select Location", "Choose your state and city. Live weather is fetched automatically."),
            ("2", "Enter Farm Details", "Set the sowing month and total cultivated area in hectares."),
            ("3", "Run Prediction", "Click Run Prediction to generate your full crop yield report."),
        ],
    ):
        with col:
            st.markdown(
                f"""
            <div class='how-card'>
                <div class='snum'>{n}</div>
                <b>{title}</b><p>{desc}</p>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Supported Crops (18)</div>", unsafe_allow_html=True)

    # ── Responsive crop grid: 6 per row on desktop, 3 on mobile ──
    # Streamlit columns auto-stack on narrow viewports
    crop_list = list(CROPS.items())
    for row_start in range(0, len(crop_list), 6):
        cols = st.columns(6)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(crop_list):
                crop_name, info = crop_list[idx]
                with col:
                    st.markdown(
                        f"""
                    <div class='crop-mini'>
                        <div class='ico'>{info['emoji']}</div>
                        <div class='cname'>{crop_name}</div>
                        <div class='msp'>MSP ₹{info['price']}/kg</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

    st.stop()


# ══════════════════════════════════════════════
#  PREDICTION ENGINE
# ══════════════════════════════════════════════
with st.spinner("Fetching weather and running prediction…"):

    if use_manual:
        temp, humidity, rain, live = manual_temp, manual_humidity, manual_rain, True
    else:
        temp, humidity, rain, live = get_weather(city)

        if temp is None:
            st.warning("⚠️ Weather API failed — please enter values manually below.")
            temp     = st.slider("Temperature (°C)", 5, 50, 28)
            humidity = st.slider("Humidity (%)", 10, 100, 60)
            rain     = st.slider("Rainfall (mm)", 0, 350, 80)
            live     = True

    best_crop, scores = analyze_crop(temp, rain, humidity, month_num)
    yield_ton, confidence = predict_yield(area, temp, rain, humidity, best_crop, state, month_num)

    if yield_ton is None:
        st.error("❌ Model failed — no prediction available")
        st.stop()

    yield_kg = yield_ton * 1000
    income, cost, profit = calc_profit(best_crop, yield_kg)
    crop_info  = CROPS[best_crop]
    weather_src = "Manual" if use_manual or not live else "Live API"


# ══════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════
st.markdown(
    f"""
<div class='crop-banner'>
    <div class='bleft'>
        <h2>Recommended Crop: {best_crop}</h2>
        <p>{state} &nbsp;·&nbsp; {MONTHS[month_num]} &nbsp;·&nbsp;
           {get_season_label(month_num)} Season &nbsp;·&nbsp; {area} ha</p>
        <span class='conf-pill'>Confidence {confidence}%</span>
        <span class='conf-pill' style='margin-left:6px;'>Weather: {weather_src}</span>
    </div>
    <div class='bright'>{crop_info['emoji']}</div>
</div>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════
#  STAT CARDS — responsive: 2-col on mobile, 5-col on desktop
# ══════════════════════════════════════════════
# Detect mobile via viewport not possible in Streamlit;
# use 5 cols which auto-wrap on narrow screens via CSS above.
c1, c2, c3, c4, c5 = st.columns(5)
for col, cls, label, val, sub in zip(
    [c1, c2, c3, c4, c5],
    ["highlight", "", "", "", ""],
    ["Predicted Yield", "Gross Income", "Net Profit", "Input Cost", "Yield / Hectare"],
    [
        f"{yield_ton:.2f} T",
        f"₹{income:,}",
        f"₹{profit:,}",
        f"₹{cost:,}",
        f"{yield_ton/area:.2f} T/ha",
    ],
    [
        f"{int(yield_kg):,} kg total",
        f"@ ₹{crop_info['price']}/kg MSP",
        "After input costs",
        "@ ₹8/kg estimate",
        f"Across {area} ha",
    ],
):
    with col:
        st.markdown(
            f"""
        <div class='stat-card {cls}'>
            <div class='label'>{label}</div>
            <div class='value'>{val}</div>
            <div class='sub'>{sub}</div>
        </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  ROW 2 — Weather | Requirements | Profit Chart
# ══════════════════════════════════════════════
cw, cr, cp = st.columns([1, 1, 1])

with cw:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Weather Conditions</div>", unsafe_allow_html=True)
    w1, w2, w3 = st.columns([1, 1, 1.3])
    w1.metric("Temp", f"{round(temp)}°C")
    w2.metric("Humidity", f"{int(humidity)}%")
    w3.metric("Rainfall", f"{int(rain)} mm")
    st.markdown(
        f"<p style='font-size:0.77rem;color:#7a9485;margin:8px 0 12px;'>"
        f"Season: <b style='color:#1a3a2a;'>{get_season_label(month_num)}</b>"
        f"&nbsp;·&nbsp;Source: <b style='color:#1a3a2a;'>{weather_src}</b></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.71rem;font-weight:700;color:#7a9485;"
        "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;'>"
        "Suitability vs Ideal Range</div>",
        unsafe_allow_html=True,
    )
    for name, (cur_val, lo, hi) in {
        "Temperature": (temp, crop_info["temp"][0], crop_info["temp"][1]),
        "Rainfall":    (rain, crop_info["rain"][0], crop_info["rain"][1]),
        "Humidity":    (humidity, crop_info["humidity"][0], crop_info["humidity"][1]),
    }.items():
        in_range = lo <= cur_val <= hi
        pct = int(np.clip((cur_val - lo) / max(hi - lo, 1) * 100, 0, 100))
        color = "#2d7a46" if in_range else "#c0392b"
        tag   = "✓ In range" if in_range else "✗ Out of range"
        st.markdown(
            f"""
        <div class='prog-wrap'>
            <div class='prog-label'>
                <span>{name} — {cur_val:.0f}
                    <span style='color:#9ab3a5;font-size:0.7rem;'>(ideal {lo}–{hi})</span>
                </span>
                <span style='color:{color};font-size:0.7rem;font-weight:600;'>{tag}</span>
            </div>
            <div class='prog-track'>
                <div class='prog-fill' style='width:{pct}%;background:{color};'></div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with cr:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-hdr'>Requirements — {best_crop}</div>", unsafe_allow_html=True
    )
    for k, v in [
        ("Temperature",  f"{crop_info['temp'][0]}–{crop_info['temp'][1]} °C"),
        ("Rainfall",     f"{crop_info['rain'][0]}–{crop_info['rain'][1]} mm"),
        ("Humidity",     f"{crop_info['humidity'][0]}–{crop_info['humidity'][1]} %"),
        ("Sowing Months", ", ".join([MONTHS[m][:3] for m in crop_info["months"][:4]])),
        ("Season",       crop_info["season"]),
        ("MSP Price",    f"₹ {crop_info['price']} / kg"),
        ("Avg Yield",    f"{get_crop_avg(best_crop):.2f} T/ha (historical)"),
    ]:
        st.markdown(
            f"""
        <div class='req-row'>
            <span class='req-key'>{k}</span>
            <span class='req-val'>{v}</span>
        </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with cp:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Revenue Breakdown</div>", unsafe_allow_html=True)

    # ── Responsive chart size ──
    fig, ax = plt.subplots(figsize=(4, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = ["Gross Income", "Input Cost", "Net Profit"]
    values = [income, cost, profit]
    colors = ["#1a3a2a", "#c0392b", "#2d7a46"]
    bars = ax.barh(labels, values, color=colors, height=0.42, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"₹{val:,}", va="center", fontsize=8.5, fontweight="600", color="#1a3a2a",
        )
    ax.set_xlim(0, max(values) * 1.32)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(labelsize=8.5, labelcolor="#5a7a6a")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)          # ← prevent memory leak on repeated runs
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  ROW 3 — Suitability Chart + Tips
# ══════════════════════════════════════════════
cc, ct = st.columns([3, 2])

with cc:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-hdr'>Crop Suitability Comparison (Top 10)</div>",
        unsafe_allow_html=True,
    )

    top10 = sorted(scores.items(), key=lambda x: -x[1])[:10]
    crop_names  = [c for c, _ in top10]
    crop_scores = [s for _, s in top10]
    bar_colors  = ["#1a3a2a" if c == best_crop else "#ccddd4" for c in crop_names]
    labels_disp = [f"{CROPS[c]['emoji']}  {c}" for c in crop_names]

    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    bars2 = ax2.bar(
        labels_disp, crop_scores, color=bar_colors,
        width=0.5, edgecolor="white", linewidth=1.2,
    )
    for bar, val in zip(bars2, crop_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, val + 1.5, str(val),
            ha="center", va="bottom", fontsize=8.5, fontweight="700",
            color="#1a3a2a" if val == max(crop_scores) else "#7a9485",
        )
    ax2.set_ylim(0, max(crop_scores) * 1.25 + 5)
    for sp in ["top", "right", "left"]: ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#e4ebe6")
    ax2.tick_params(bottom=False, left=False)
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_tick_params(labelsize=8, labelcolor="#3a5c47", rotation=15)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)         # ← prevent memory leak
    st.markdown("</div>", unsafe_allow_html=True)

with ct:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-hdr'>Agronomic Tips — {best_crop}</div>", unsafe_allow_html=True
    )
    for tip in crop_info.get("tips", []):
        st.markdown(
            f"""
        <div class='tip-row'>
            <div class='tip-dot'></div>
            <div>{tip}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  FULL TABLE — All 18 crops
# ══════════════════════════════════════════════
st.markdown(
    "<div class='section-hdr'>All Crops — Detailed Score Breakdown</div>",
    unsafe_allow_html=True,
)

rows = []
for crop, score in sorted(scores.items(), key=lambda x: -x[1]):
    info = CROPS[crop]
    suit = (
        "Excellent" if score >= 70 else
        "Good"      if score >= 50 else
        "Moderate"  if score >= 30 else "Poor"
    )
    rows.append({
        "Crop":           f"{info['emoji']}  {crop}",
        "Score / 100":    score,
        "Suitability":    suit,
        "Season":         info["season"],
        "Temp Range":     f"{info['temp'][0]}–{info['temp'][1]} °C",
        "Rainfall":       f"{info['rain'][0]}–{info['rain'][1]} mm",
        "Humidity":       f"{info['humidity'][0]}–{info['humidity'][1]} %",
        "MSP ₹/kg":       info["price"],
        "Avg Yield T/ha": round(get_crop_avg(crop), 2),
    })

st.dataframe(
    pd.DataFrame(rows).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)


# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown(
    f"""
<div class='footer-bar'>
    <span>Crop Yield Prediction System &nbsp;·&nbsp; GradientBoosting Regressor</span>
    <span>Training Data: Govt. of India · 246,091+ records &nbsp;·&nbsp;
          Weather: OpenWeatherMap ({weather_src})</span>
    <span>Predictions are indicative. Actual yield depends on soil, irrigation, and practices.</span>
</div>
""",
    unsafe_allow_html=True,
)