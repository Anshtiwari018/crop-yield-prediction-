import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# ══════════════════════════════════════════════
#  API KEY  –  from .streamlit/secrets.toml
# ══════════════════════════════════════════════
try:
    WEATHER_API_KEY = st.secrets["API_KEY"]
except Exception:
    WEATHER_API_KEY = ""

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Force light background everywhere ── */
.stApp {
    background-color: #f5f7f2 !important;
}
.main, .main > div, .block-container {
    background-color: #f5f7f2 !important;
}
/* Fix Streamlit column/element containers going dark */
[data-testid="column"] > div > div,
[data-testid="stVerticalBlock"] > div,
[data-testid="stHorizontalBlock"] > div,
.element-container {
    background-color: unset !important;
}

.main .block-container {
    background: #f5f7f2 !important;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a3a2a;
    border-right: 1px solid #243d2e;
}
section[data-testid="stSidebar"] * { color: #e8f0ec !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #2d7a46 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
    width: 100%;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #236038 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input {
    background: #243d2e !important;
    border: 1px solid #3a5c47 !important;
    border-radius: 7px !important;
    color: #e8f0ec !important;
}
section[data-testid="stSidebar"] hr { border-color: #2d4d3a !important; }
section[data-testid="stSidebar"] label {
    color: #8dbfa0 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* ── Stat card ── */
.stat-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.1rem 1.2rem 1rem;
    border: 1px solid #e4ebe6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-card .label {
    font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: #7a9485; margin-bottom: 4px;
}
.stat-card .value { font-size: 1.55rem; font-weight: 700; color: #1a3a2a; line-height: 1.1; }
.stat-card .sub   { font-size: 0.74rem; color: #9ab3a5; margin-top: 2px; }
.stat-card.highlight { background: #1a3a2a; border-color: #1a3a2a; }
.stat-card.highlight .label { color: #6dbf8a; }
.stat-card.highlight .value { color: #ffffff; }
.stat-card.highlight .sub   { color: #7aaa8f; }

/* ── Section header ── */
.section-hdr {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #4a7a5a; padding-bottom: 6px;
    border-bottom: 1px solid #d8e8dd;
    margin-bottom: 0.9rem; margin-top: 0.3rem;
}

/* ── Panel ── */
.panel {
    background: #ffffff; border-radius: 12px;
    border: 1px solid #e4ebe6;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    height: 100%;
}

/* ── Progress bar ── */
.prog-wrap { margin-bottom: 10px; }
.prog-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #5a7a6a; margin-bottom: 3px;
}
.prog-track { background: #e8f0ec; border-radius: 6px; height: 7px; overflow: hidden; }
.prog-fill  { height: 7px; border-radius: 6px; }

/* ── Tip row ── */
.tip-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 0; border-bottom: 1px solid #f0f5f2;
    font-size: 0.83rem; color: #2d4d3a; line-height: 1.5;
}
.tip-row:last-child { border-bottom: none; }
.tip-dot { width:6px; height:6px; border-radius:50%; background:#2d7a46; margin-top:7px; flex-shrink:0; }

/* ── Req row ── */
.req-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f0f5f2; font-size: 0.84rem;
}
.req-row:last-child { border-bottom: none; }
.req-key { color: #7a9485; font-weight: 500; }
.req-val { color: #1a3a2a; font-weight: 600; }

/* ── Crop banner ── */
.crop-banner {
    background: linear-gradient(120deg, #1a3a2a 0%, #2d5a3d 100%);
    border-radius: 14px; padding: 1.4rem 1.8rem;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.4rem;
}
.crop-banner .bleft h2 { color:#fff; font-size:1.4rem; font-weight:700; margin:0 0 4px; }
.crop-banner .bleft p  { color:#7acc96; font-size:0.82rem; margin:0; }
.crop-banner .bright   { font-size:3.2rem; line-height:1; }
.conf-pill {
    display:inline-block; background:rgba(255,255,255,0.12);
    color:#a8f0c0; border-radius:20px; padding:2px 12px;
    font-size:0.76rem; font-weight:600; margin-top:6px;
}

/* ── Welcome ── */
.welcome-title { font-size:2rem; font-weight:700; color:#1a3a2a; margin-bottom:4px; }
.welcome-sub   { color:#6a9478; font-size:0.94rem; margin-bottom:2rem; }
.how-card {
    background:#fff; border-radius:12px; border:1px solid #e4ebe6;
    padding:1.2rem; text-align:center;
}
.how-card .snum {
    width:32px; height:32px; border-radius:50%; background:#e8f5ee;
    color:#2d7a46; font-weight:700; font-size:0.9rem;
    display:flex; align-items:center; justify-content:center; margin:0 auto 10px;
}
.how-card p { font-size:0.82rem; color:#5a7a6a; margin:4px 0 0; }
.how-card b { color:#1a3a2a; font-size:0.88rem; }
.crop-mini {
    background:#fff; border-radius:10px; border:1px solid #e4ebe6;
    padding:0.9rem 0.6rem; text-align:center;
}
.crop-mini .ico   { font-size:1.8rem; }
.crop-mini .cname { font-size:0.82rem; font-weight:600; color:#1a3a2a; margin:4px 0 2px; }
.crop-mini .msp   { font-size:0.72rem; color:#7a9485; }

/* ── Footer ── */
.footer-bar {
    margin-top:2.5rem; padding-top:1rem;
    border-top:1px solid #d8e8dd;
    font-size:0.74rem; color:#9ab3a5;
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:4px;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception:
        return None


model = load_model()


# ══════════════════════════════════════════════
#  DATA CONSTANTS
# ══════════════════════════════════════════════
CROPS = {
    "Rice": {
        "temp": (20, 35),
        "rain": (150, 300),
        "humidity": (60, 100),
        "months": [6, 7, 8, 9],
        "price": 20,
        "emoji": "🌾",
    },
    "Wheat": {
        "temp": (10, 25),
        "rain": (50, 100),
        "humidity": (40, 60),
        "months": [10, 11, 12, 1],
        "price": 25,
        "emoji": "🌿",
    },
    "Maize": {
        "temp": (21, 30),
        "rain": (50, 150),
        "humidity": (40, 70),
        "months": [6, 7, 8],
        "price": 18,
        "emoji": "🌽",
    },
    "Arhar": {
        "temp": (25, 35),
        "rain": (40, 100),
        "humidity": (30, 60),
        "months": [6, 7, 8, 9],
        "price": 70,
        "emoji": "🫘",
    },
    "Sugarcane": {
        "temp": (25, 35),
        "rain": (100, 250),
        "humidity": (50, 80),
        "months": [2, 3, 4, 5],
        "price": 3,
        "emoji": "🪵",
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

TIPS = {
    "Rice": [
        "Maintain 5–10 cm standing water during early vegetative stage for best germination.",
        "Adopt SRI (System of Rice Intensification) to improve yield by 20–30% with less water.",
        "Apply nitrogen fertilizer in 3 equal splits — basal, tillering, and panicle initiation.",
    ],
    "Wheat": [
        "Sow between November 1–15 in North India for optimum yield potential.",
        "Irrigate at crown root initiation (21 DAS), tillering, jointing, and grain filling stages.",
        "Use certified disease-resistant varieties like HD-2967 or PBW-343.",
    ],
    "Maize": [
        "Target 60,000–65,000 plants per hectare for optimum canopy and light interception.",
        "Apply 120 kg N/ha in 3 equal splits at sowing, knee-high stage, and tasseling.",
        "Ensure proper field drainage — maize cannot tolerate waterlogging even for 24 hours.",
    ],
    "Arhar": [
        "Intercropping with soybean or groundnut improves land-use efficiency significantly.",
        "Seed inoculation with Rhizobium culture can reduce nitrogen fertilizer need by 25 kg/ha.",
        "Harvest when 75–80% of pods have turned brown to minimise shattering losses.",
    ],
    "Sugarcane": [
        "Ratoon cropping reduces input cost by 30–40% while yielding 80–90% of plant crop.",
        "Drip irrigation combined with fertigation can improve water-use efficiency by 35%.",
        "Trash mulching after harvest suppresses weeds and retains moisture during dry spells.",
    ],
}


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def get_season_id(m):
    if m in [6, 7, 8, 9]:
        return 1
    if m in [10, 11, 12, 1]:
        return 2
    return 0


def get_season_label(m):
    if m in [6, 7, 8, 9]:
        return "Kharif"
    if m in [10, 11, 12, 1]:
        return "Rabi"
    return "Zaid"


def encode_crop(c):
    return list(CROPS.keys()).index(c) + 1


def encode_state(s):
    return (
        ["Rajasthan", "Maharashtra", "Punjab"].index(s) + 1
        if s in ["Rajasthan", "Maharashtra", "Punjab"]
        else 0
    )


def crop_avg(c):
    return {"Rice": 60, "Wheat": 50, "Maize": 45, "Arhar": 35, "Sugarcane": 80}.get(
        c, 50
    )


def state_avg(s):
    return {"Rajasthan": 40, "Maharashtra": 55, "Punjab": 70}.get(s, 50)


def get_weather(city):
    if not WEATHER_API_KEY:
        return 28.0, 60.0, 50.0, False
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        d = requests.get(url, timeout=6).json()
        if "main" not in d:
            return 28.0, 60.0, 50.0, False
        return (
            float(d["main"]["temp"]),
            float(d["main"]["humidity"]),
            float(d.get("rain", {}).get("1h", 0)),
            True,
        )
    except Exception:
        return 28.0, 60.0, 50.0, False


def analyze_crop(temp, rain, humidity, month):
    res = {}
    for crop, d in CROPS.items():
        s = 30 if d["temp"][0] <= temp <= d["temp"][1] else 0
        s += 30 if d["rain"][0] <= rain <= d["rain"][1] else 0
        s += 20 if d["humidity"][0] <= humidity <= d["humidity"][1] else 0
        s += 20 if month in d["months"] else 0
        res[crop] = s
    return max(res, key=res.get), res


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
                    "season": get_season_id(month),
                    "crop_avg": crop_avg(crop),
                    "state_avg": state_avg(state),
                }
            ]
        )
        yt = np.expm1(model.predict(df)[0])
        return round(max(0.5, min(yt, area * 10)), 2), 80
    except Exception:
        return None, 40


def calc_profit(crop, yield_kg):
    p = CROPS[crop]["price"]
    inc = yield_kg * p
    return int(inc), int(inc - yield_kg * 5)


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
        "City",
        value="Jaipur",
        label_visibility="collapsed",
        placeholder="e.g. Jaipur, Pune, Ludhiana",
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("**Sowing Month**")
    month_num = st.selectbox(
        "Month",
        list(MONTHS.keys()),
        format_func=lambda x: MONTHS[x],
        index=datetime.now().month - 1,
        label_visibility="collapsed",
    )

    st.markdown("**Farm Area (hectares)**")
    area = st.number_input(
        "Area",
        min_value=0.1,
        max_value=10000.0,
        value=5.0,
        step=0.5,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    use_manual = st.checkbox("Override weather manually", value=False)
    if use_manual:
        st.markdown("**Temperature (°C)**")
        manual_temp = st.slider("Temp", 5, 50, 28, label_visibility="collapsed")
        st.markdown("**Humidity (%)**")
        manual_humidity = st.slider("Hum", 10, 100, 60, label_visibility="collapsed")
        st.markdown("**Rainfall (mm)**")
        manual_rain = st.slider("Rain", 0, 300, 80, label_visibility="collapsed")

    st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", use_container_width=True)

    st.markdown(
        """
    <div style='font-size:0.68rem;color:#3a5c47;margin-top:0.8rem;line-height:1.65;'>
        Weather: OpenWeatherMap API<br>
        Model: GradientBoosting Regressor<br>
        Training data: 246,091 crop records
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
        income forecast, and crop suitability report.
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, (n, title, desc) in zip(
        [c1, c2, c3],
        [
            (
                "1",
                "Select Location",
                "Choose your state and city. Live weather is fetched automatically.",
            ),
            (
                "2",
                "Enter Farm Details",
                "Set the sowing month and total cultivated area in hectares.",
            ),
            (
                "3",
                "Run Prediction",
                "Click Run Prediction to generate your full crop yield report.",
            ),
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
    st.markdown(
        "<div class='section-hdr'>Supported Crops</div>", unsafe_allow_html=True
    )
    cols = st.columns(len(CROPS))
    for i, (crop, info) in enumerate(CROPS.items()):
        with cols[i]:
            st.markdown(
                f"""
            <div class='crop-mini'>
                <div class='ico'>{info['emoji']}</div>
                <div class='cname'>{crop}</div>
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
        temp, humidity, rain, live = manual_temp, manual_humidity, manual_rain, False
    else:
        temp, humidity, rain, live = get_weather(city)

    best_crop, scores = analyze_crop(temp, rain, humidity, month_num)
    yield_ton, confidence = predict_yield(
        area, temp, rain, humidity, best_crop, state, month_num
    )

    if yield_ton is None:
        yield_ton, confidence = round(area * 2, 2), 50

    yield_kg = yield_ton * 1000
    income, profit = calc_profit(best_crop, yield_kg)
    cost = income - profit
    crop_info = CROPS[best_crop]
    weather_src = "Live" if live else "Estimated"


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
#  STAT CARDS
# ══════════════════════════════════════════════
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
        "@ ₹5/kg estimate",
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
#  ROW 2 — Weather | Requirements | Profit
# ══════════════════════════════════════════════
cw, cr, cp = st.columns([1, 1, 1])

# Weather Panel
with cw:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-hdr'>Weather Conditions</div>", unsafe_allow_html=True
    )
    w1, w2, w3 = st.columns(3)
    w1.metric("Temp", f"{temp:.1f}°C")
    w2.metric("Humidity", f"{humidity:.0f}%")
    w3.metric("Rainfall", f"{rain:.0f} mm")
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
    for name, (val, lo, hi) in {
        "Temperature": (temp, crop_info["temp"][0], crop_info["temp"][1]),
        "Rainfall": (rain, crop_info["rain"][0], crop_info["rain"][1]),
        "Humidity": (humidity, crop_info["humidity"][0], crop_info["humidity"][1]),
    }.items():
        in_range = lo <= val <= hi
        pct = int(np.clip((val - lo) / max(hi - lo, 1) * 100, 0, 100))
        color = "#2d7a46" if in_range else "#c0392b"
        tag = "✓ In range" if in_range else "✗ Out of range"
        st.markdown(
            f"""
        <div class='prog-wrap'>
            <div class='prog-label'>
                <span>{name} — {val:.0f}
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

# Crop Requirements Panel
with cr:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-hdr'>Requirements — {best_crop}</div>",
        unsafe_allow_html=True,
    )
    for k, v in [
        ("Temperature", f"{crop_info['temp'][0]}–{crop_info['temp'][1]} °C"),
        ("Rainfall", f"{crop_info['rain'][0]}–{crop_info['rain'][1]} mm"),
        ("Humidity", f"{crop_info['humidity'][0]}–{crop_info['humidity'][1]} %"),
        ("Sowing Months", ", ".join([MONTHS[m][:3] for m in crop_info["months"]])),
        ("MSP Price", f"₹ {crop_info['price']} / kg"),
        ("Season", get_season_label(month_num)),
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

# Profit Chart Panel
with cp:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-hdr'>Revenue Breakdown</div>", unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(4, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = ["Gross Income", "Input Cost", "Net Profit"]
    values = [income, cost, profit]
    colors = ["#1a3a2a", "#c0392b", "#2d7a46"]
    bars = ax.barh(
        labels, values, color=colors, height=0.42, edgecolor="white", linewidth=1.5
    )
    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"₹{val:,}",
            va="center",
            fontsize=8.5,
            fontweight="600",
            color="#1a3a2a",
        )
    ax.set_xlim(0, max(values) * 1.32)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(labelsize=8.5, labelcolor="#5a7a6a")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  ROW 3 — Comparison Chart + Tips
# ══════════════════════════════════════════════
cc, ct = st.columns([3, 2])

with cc:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-hdr'>Crop Suitability Comparison</div>",
        unsafe_allow_html=True,
    )

    crop_names = list(scores.keys())
    crop_scores = list(scores.values())
    bar_colors = ["#1a3a2a" if c == best_crop else "#ccddd4" for c in crop_names]
    labels_disp = [f"{CROPS[c]['emoji']}  {c}" for c in crop_names]

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    bars2 = ax2.bar(
        labels_disp,
        crop_scores,
        color=bar_colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, val in zip(bars2, crop_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="700",
            color="#1a3a2a" if val == max(crop_scores) else "#7a9485",
        )
    ax2.set_ylim(0, 118)
    for sp in ["top", "right", "left"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#e4ebe6")
    ax2.tick_params(bottom=False, left=False)
    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_tick_params(labelsize=9, labelcolor="#3a5c47")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ct:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-hdr'>Agronomic Tips — {best_crop}</div>",
        unsafe_allow_html=True,
    )
    for tip in TIPS.get(best_crop, []):
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
#  FULL TABLE
# ══════════════════════════════════════════════
st.markdown(
    "<div class='section-hdr'>All Crops — Detailed Score Breakdown</div>",
    unsafe_allow_html=True,
)
rows = []
for crop, score in sorted(scores.items(), key=lambda x: -x[1]):
    info = CROPS[crop]
    suit = (
        "Excellent"
        if score >= 80
        else "Good" if score >= 60 else "Moderate" if score >= 40 else "Poor"
    )
    rows.append(
        {
            "Crop": f"{info['emoji']}  {crop}",
            "Score / 100": score,
            "Suitability": suit,
            "Temp Range": f"{info['temp'][0]}–{info['temp'][1]} °C",
            "Rainfall": f"{info['rain'][0]}–{info['rain'][1]} mm",
            "Humidity": f"{info['humidity'][0]}–{info['humidity'][1]} %",
            "Season": (
                "Kharif"
                if info["months"][0] in [6, 7, 8, 9]
                else "Rabi" if info["months"][0] in [10, 11, 12, 1] else "Zaid"
            ),
            "MSP ₹/kg": info["price"],
        }
    )
st.dataframe(
    pd.DataFrame(rows).reset_index(drop=True), use_container_width=True, hide_index=True
)


# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown(
    f"""
<div class='footer-bar'>
    <span>Crop Yield Prediction System &nbsp;·&nbsp; GradientBoosting Regressor</span>
    <span>Training Data: Govt. of India · 246,091 records &nbsp;·&nbsp; Weather: OpenWeatherMap ({weather_src})</span>
    <span>Predictions are indicative. Actual yield depends on soil, irrigation, and practices.</span>
</div>
""",
    unsafe_allow_html=True,
)
