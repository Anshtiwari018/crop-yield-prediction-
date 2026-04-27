import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from main import run_prediction

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CropIQ – crop yield prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets.get("API_KEY", None)

STATES = {
    "Rajasthan": ["Jaipur", "Kota", "Udaipur", "Jodhpur", "Ajmer"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
    "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur"],
}

CROP_EMOJI = {
    "Rice": "🌾", "Wheat": "🌿", "Maize": "🌽",
    "Arhar": "🫘", "Sugarcane": "🎋",
}

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ============================================================
#  GLOBAL CSS  ─ dark agri-tech with editorial typography
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── ROOT PALETTE ──────────────────────────────────────────── */
:root {
    --bg0: #080c10;
    --bg1: #0e1419;
    --bg2: #141b23;
    --bg3: #1c2530;
    --bg4: #232e3c;
    --acc-green: #3ddc84;
    --acc-teal:  #00c9b1;
    --acc-gold:  #f5c842;
    --acc-red:   #f05252;
    --acc-blue:  #4d9de0;
    --t1: #e8f0f7;
    --t2: #8fa3b8;
    --t3: #4d6070;
    --border: rgba(255,255,255,0.07);
    --radius: 14px;
    --radius-sm: 8px;
}

/* ── STREAMLIT SHELL ───────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg0) !important;
    color: var(--t1) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--t1) !important; }

/* hide streamlit default decoration */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── SIDEBAR ───────────────────────────────────────────────── */
.sidebar-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    letter-spacing: 3px;
    background: linear-gradient(120deg, var(--acc-green), var(--acc-teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 4px;
}
.sidebar-tagline {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--t3);
    margin-bottom: 28px;
}
.sidebar-section {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--t3);
    padding: 14px 0 6px;
    border-top: 1px solid var(--border);
}

/* ── RADIO NAV ─────────────────────────────────────────────── */
[data-testid="stRadio"] > label { display: none; }
[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
[data-testid="stRadio"] > div > label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-sm) !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--t2) !important;
    cursor: pointer;
    transition: all .2s;
}
[data-testid="stRadio"] > div > label:hover {
    background: var(--bg3) !important;
    color: var(--t1) !important;
}
[data-testid="stRadio"] > div > label[data-baseweb="radio"]:has(input:checked),
[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(61,220,132,.12) !important;
    border-color: rgba(61,220,132,.3) !important;
    color: var(--acc-green) !important;
}

/* ── SELECTS & INPUTS ──────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--t1) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── SLIDERS ───────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--acc-green) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--acc-green) !important;
    border: 2px solid var(--bg0) !important;
    box-shadow: 0 0 0 3px rgba(61,220,132,.25) !important;
}

/* ── PRIMARY BUTTON ────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(120deg, var(--acc-green), var(--acc-teal)) !important;
    color: #060d0a !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 32px !important;
    letter-spacing: .5px;
    transition: opacity .2s, transform .15s !important;
    box-shadow: 0 4px 20px rgba(61,220,132,.25) !important;
}
[data-testid="stButton"] > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active { transform: scale(.98) !important; }

/* ── SUCCESS / INFO BANNERS ────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(61,220,132,.08) !important;
    border: 1px solid rgba(61,220,132,.25) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--acc-green) !important;
}

/* ── PROGRESS BAR ──────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--acc-green), var(--acc-teal)) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: var(--bg3) !important;
    border-radius: 4px !important;
}

/* ── DATAFRAME ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── SPINNER ───────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: var(--acc-green) !important; }

/* ── CUSTOM COMPONENTS ─────────────────────────────────────── */

/* Page heading */
.phead {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    letter-spacing: 3px;
    line-height: 1;
    background: linear-gradient(120deg, var(--acc-green) 0%, var(--acc-teal) 60%, var(--t1) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.phead-sub {
    font-size: 13px;
    color: var(--t3);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 28px;
}

/* Section label */
.slabel {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: var(--t3);
    font-weight: 600;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* KPI card */
.kpi {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    animation: fadein .5s ease both;
}
.kpi::after {
    content: '';
    position: absolute;
    bottom: -20px; right: -20px;
    width: 80px; height: 80px;
    border-radius: 50%;
    opacity: .06;
}
.kpi-green::after  { background: var(--acc-green); }
.kpi-teal::after   { background: var(--acc-teal); }
.kpi-gold::after   { background: var(--acc-gold); }
.kpi-blue::after   { background: var(--acc-blue); }

.kpi-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--t3);
    font-weight: 600;
    margin-bottom: 10px;
}
.kpi-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 38px;
    letter-spacing: 1px;
    line-height: 1;
    color: var(--t1);
}
.kpi-unit {
    font-size: 12px;
    color: var(--t2);
    margin-top: 4px;
}
.kpi-icon {
    position: absolute;
    top: 18px; right: 18px;
    font-size: 22px;
    opacity: .35;
}

/* Weather pill row */
.wx-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 4px;
}
.wx-pill {
    flex: 1;
    min-width: 100px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    text-align: center;
}
.wx-icon  { font-size: 24px; margin-bottom: 6px; }
.wx-val   { font-family: 'Bebas Neue', sans-serif; font-size: 28px; letter-spacing: 1px; color: var(--t1); }
.wx-lbl   { font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--t3); margin-top: 3px; }

/* Crop highlight card */
.crop-card {
    background: linear-gradient(135deg, rgba(61,220,132,.08), rgba(0,201,177,.05));
    border: 1px solid rgba(61,220,132,.2);
    border-radius: var(--radius);
    padding: 28px;
    text-align: center;
    animation: fadein .6s ease both;
}
.crop-emoji { font-size: 56px; margin-bottom: 10px; }
.crop-name  {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 48px;
    letter-spacing: 3px;
    color: var(--acc-green);
    line-height: 1;
}
.crop-sub { font-size: 12px; color: var(--t3); text-transform: uppercase; letter-spacing: 2px; margin-top: 6px; }

/* Crop detail table */
.ctable {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
}
.ctable tr { border-bottom: 1px solid var(--border); }
.ctable tr:last-child { border-bottom: none; }
.ctable td { padding: 10px 4px; color: var(--t2); }
.ctable td:last-child { text-align: right; color: var(--t1); font-weight: 500; }

/* Confidence bar */
.conf-wrap { margin-top: 6px; }
.conf-track {
    background: var(--bg3);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin-top: 8px;
}
.conf-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--acc-green), var(--acc-teal));
    transition: width .9s cubic-bezier(.22,1,.36,1);
}
.conf-meta { display: flex; justify-content: space-between; font-size: 11px; color: var(--t3); margin-top: 5px; }

/* Profit highlight */
.profit-highlight {
    background: rgba(245,200,66,.06);
    border: 1px solid rgba(245,200,66,.2);
    border-radius: var(--radius-sm);
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 12px;
}
.ph-label { font-size: 12px; color: var(--t2); text-transform: uppercase; letter-spacing: 1px; }
.ph-value { font-family: 'Bebas Neue', sans-serif; font-size: 32px; color: var(--acc-gold); letter-spacing: 1px; }

/* About page */
.about-hero {
    background: linear-gradient(135deg, rgba(61,220,132,.06), rgba(0,201,177,.04));
    border: 1px solid rgba(61,220,132,.15);
    border-radius: var(--radius);
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 24px;
}
.about-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 56px;
    letter-spacing: 5px;
    background: linear-gradient(120deg, var(--acc-green), var(--acc-teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.about-desc { font-size: 15px; color: var(--t2); line-height: 1.7; max-width: 560px; margin: 16px auto 0; }

.feature-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.feature-item {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 18px 20px;
}
.fi-icon { font-size: 24px; margin-bottom: 8px; }
.fi-title { font-weight: 600; color: var(--t1); margin-bottom: 4px; }
.fi-desc { font-size: 12px; color: var(--t2); line-height: 1.5; }

/* Animations */
@keyframes fadein {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate { animation: fadein .5s ease both; }
.delay1  { animation-delay: .1s; }
.delay2  { animation-delay: .2s; }
.delay3  { animation-delay: .3s; }

/* column gap fix */
[data-testid="column"] { padding: 0 6px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  HELPERS
# ============================================================
def load_history():
    if os.path.exists("history.csv"):
        return pd.read_csv("history.csv", on_bad_lines="skip")
    return pd.DataFrame()


def save_history(entry: dict):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv("history.csv", index=False)


def kpi_card(label: str, value, unit: str, icon: str, color: str):
    st.markdown(f"""
    <div class="kpi kpi-{color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="slabel">{title}</div>', unsafe_allow_html=True)


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">CropIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Smart Farming Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["🌾  Dashboard", "📊  Analytics", "📜  History", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">Quick Stats</div>', unsafe_allow_html=True)
    df_hist = load_history()
    total_runs = len(df_hist)
    avg_yield = round(df_hist["Yield"].mean(), 2) if not df_hist.empty and "Yield" in df_hist.columns else "–"
    st.markdown(f"""
    <div style="display:flex;gap:10px;margin-top:4px">
        <div style="flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:24px;color:var(--acc-green)">{total_runs}</div>
            <div style="font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:1px">Predictions</div>
        </div>
        <div style="flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:24px;color:var(--acc-teal)">{avg_yield}</div>
            <div style="font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:1px">Avg Yield T</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:var(--t3);text-align:center;line-height:1.6;padding:12px;
                background:var(--bg2);border:1px solid var(--border);border-radius:8px">
        Powered by RandomForest ML<br>+ OpenWeatherMap API
    </div>""", unsafe_allow_html=True)


# ============================================================
#  DASHBOARD
# ============================================================
if "Dashboard" in menu:

    st.markdown('<div class="phead">crop yield prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="phead-sub">AI-powered yield prediction & crop recommendation</div>', unsafe_allow_html=True)

    # ── INPUT PANEL ──────────────────────────────────────────
    section("Input Parameters")
    with st.container():
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
        with c1:
            state = st.selectbox("State", list(STATES.keys()), label_visibility="visible")
        with c2:
            city = st.selectbox("City / District", STATES[state], label_visibility="visible")
        with c3:
            area = st.slider("Area (Hectare)", 1, 500, 50)
        with c4:
            month_idx = st.slider("Month", 1, 12, datetime.now().month)
            st.caption(f"Selected: **{MONTH_NAMES[month_idx]}**")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    run_col, _ = st.columns([1, 4])
    with run_col:
        predict = st.button("⚡  Run Prediction", use_container_width=True)

    # ── RESULTS ──────────────────────────────────────────────
    if predict:
        with st.spinner("Analyzing weather & running ML model…"):
            result = run_prediction(area, city, month_idx, API_KEY)

        st.success("✓ Prediction complete")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # KPI ROW
        section("Key Metrics")
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("Yield", result["yield_ton"], "metric tons", "🌾", "green")
        with k2: kpi_card("Production", f"{result['yield_kg']:,}", "kilograms", "📦", "teal")
        with k3: kpi_card("Gross Income", f"₹{result['income']:,}", "estimated", "💰", "gold")
        with k4: kpi_card("Net Profit", f"₹{result['profit']:,}", "after costs", "📈", "blue")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # MAIN CONTENT ROW
        left, right = st.columns([1.55, 1])

        with left:
            # PERFORMANCE CHART
            section("Performance Overview")
            df_bar = pd.DataFrame({
                "Metric": ["Yield (T)", "Income (K₹)", "Profit (K₹)"],
                "Value":  [result["yield_ton"],
                           round(result["income"]/1000, 1),
                           round(result["profit"]/1000, 1)],
                "Color":  ["#3ddc84", "#00c9b1", "#f5c842"],
            })
            fig_bar = go.Figure(go.Bar(
                x=df_bar["Metric"], y=df_bar["Value"],
                marker_color=df_bar["Color"],
                text=df_bar["Value"], textposition="outside",
                textfont=dict(color="#8fa3b8", size=12),
                width=0.45,
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=240,
                font=dict(family="DM Sans", color="#8fa3b8"),
                xaxis=dict(showgrid=False, tickfont=dict(size=12)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                           zeroline=False, tickfont=dict(size=11)),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # WEATHER SECTION
            section("Live Weather Data")
            st.markdown(f"""
            <div class="wx-row animate">
                <div class="wx-pill">
                    <div class="wx-icon">🌡</div>
                    <div class="wx-val">{result['temp']}°</div>
                    <div class="wx-lbl">Temperature (C)</div>
                </div>
                <div class="wx-pill">
                    <div class="wx-icon">🌧</div>
                    <div class="wx-val">{result['rain']}</div>
                    <div class="wx-lbl">Rainfall (mm)</div>
                </div>
                <div class="wx-pill">
                    <div class="wx-icon">💧</div>
                    <div class="wx-val">{result['humidity']}%</div>
                    <div class="wx-lbl">Humidity</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # CONFIDENCE
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            section("Model Confidence")
            conf = result["confidence"]
            conf_color = "var(--acc-green)" if conf >= 75 else "var(--acc-gold)" if conf >= 55 else "var(--acc-red)"
            st.markdown(f"""
            <div class="conf-wrap animate">
                <div style="display:flex;justify-content:space-between;align-items:baseline">
                    <span style="font-size:13px;color:var(--t2)">Prediction reliability</span>
                    <span style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:{conf_color}">{conf}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{conf}%"></div>
                </div>
                <div class="conf-meta">
                    <span>Low confidence</span><span>High confidence</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with right:
            # RECOMMENDED CROP
            section("Recommended Crop")
            emoji = CROP_EMOJI.get(result["crop"], "🌱")
            st.markdown(f"""
            <div class="crop-card">
                <div class="crop-emoji">{emoji}</div>
                <div class="crop-name">{result["crop"]}</div>
                <div class="crop-sub">Best match for current conditions</div>
            </div>""", unsafe_allow_html=True)

            # CROP DETAILS
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            section("Ideal Growing Conditions")
            d = result["details"]
            months_str = " · ".join(MONTH_NAMES[m] for m in d["months"] if m)
            st.markdown(f"""
            <div class="kpi" style="animation:none">
                <table class="ctable">
                    <tr><td>Temperature</td><td>{d['temp']}</td></tr>
                    <tr><td>Rainfall</td><td>{d['rain']}</td></tr>
                    <tr><td>Humidity</td><td>{d['humidity']}</td></tr>
                    <tr><td>Best Months</td><td>{months_str}</td></tr>
                </table>
            </div>""", unsafe_allow_html=True)

            # PROFIT HIGHLIGHT
            st.markdown(f"""
            <div class="profit-highlight animate">
                <div>
                    <div class="ph-label">Estimated Profit</div>
                    <div style="font-size:11px;color:var(--t3);margin-top:3px">After cultivation cost</div>
                </div>
                <div class="ph-value">₹{result['profit']:,}</div>
            </div>""", unsafe_allow_html=True)

        # SAVE TO HISTORY
        save_history({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "State": state,
            "City": city,
            "Area_Ha": area,
            "Month": MONTH_NAMES[month_idx],
            "Crop": result["crop"],
            "Yield": result["yield_ton"],
            "Income": result["income"],
            "Profit": result["profit"],
            "Confidence": result["confidence"],
        })

    else:
        # Placeholder when no prediction yet
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:var(--t3)">
            <div style="font-size:52px;margin-bottom:16px;opacity:.3">🌾</div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:22px;letter-spacing:3px;color:var(--t3)">
                Configure inputs above and run prediction
            </div>
        </div>""", unsafe_allow_html=True)


# ============================================================
#  ANALYTICS
# ============================================================
elif "Analytics" in menu:

    st.markdown('<div class="phead">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="phead-sub">Historical yield trends & crop distribution</div>', unsafe_allow_html=True)

    df = load_history()

    if df.empty:
        st.info("No prediction history found. Run a prediction first.")
    else:
        # Summary KPIs
        section("Summary")
        s1, s2, s3, s4 = st.columns(4)
        with s1: kpi_card("Total Runs", len(df), "predictions", "🔢", "green")
        with s2: kpi_card("Avg Yield", round(df["Yield"].mean(), 2) if "Yield" in df.columns else "–", "metric tons", "🌾", "teal")
        with s3: kpi_card("Best Yield", round(df["Yield"].max(), 2) if "Yield" in df.columns else "–", "metric tons", "🏆", "gold")
        with s4:
            top_crop = df["Crop"].mode()[0] if "Crop" in df.columns and not df["Crop"].empty else "–"
            kpi_card("Top Crop", top_crop, "most predicted", "🌱", "blue")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1.6, 1])

        with col_a:
            section("Yield Trend Over Time")
            if "Date" in df.columns and "Yield" in df.columns:
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=df["Date"], y=df["Yield"],
                    mode="lines+markers",
                    line=dict(color="#3ddc84", width=2),
                    marker=dict(color="#3ddc84", size=6),
                    fill="tozeroy",
                    fillcolor="rgba(61,220,132,0.07)",
                    name="Yield (T)",
                ))
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0), height=260,
                    font=dict(family="DM Sans", color="#8fa3b8"),
                    xaxis=dict(showgrid=False, tickfont=dict(size=11)),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            section("Crop Distribution")
            if "Crop" in df.columns:
                crop_counts = df["Crop"].value_counts().reset_index()
                crop_counts.columns = ["Crop", "Count"]
                fig_pie = go.Figure(go.Pie(
                    labels=crop_counts["Crop"], values=crop_counts["Count"],
                    marker=dict(colors=["#3ddc84","#00c9b1","#f5c842","#4d9de0","#bc8cff"]),
                    hole=0.55,
                    textinfo="label+percent",
                    textfont=dict(size=11, color="#8fa3b8"),
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=260, showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        # Income vs Profit
        if "Income" in df.columns and "Profit" in df.columns:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            section("Income vs Profit Comparison")
            fig_dual = go.Figure()
            fig_dual.add_trace(go.Bar(x=df["Date"], y=df["Income"],
                name="Income", marker_color="#00c9b1", opacity=0.85))
            fig_dual.add_trace(go.Bar(x=df["Date"], y=df["Profit"],
                name="Profit", marker_color="#f5c842", opacity=0.85))
            fig_dual.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0), height=240, barmode="group",
                font=dict(family="DM Sans", color="#8fa3b8"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
            )
            st.plotly_chart(fig_dual, use_container_width=True, config={"displayModeBar": False})


# ============================================================
#  HISTORY
# ============================================================
elif "History" in menu:

    st.markdown('<div class="phead">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="phead-sub">Complete log of all past predictions</div>', unsafe_allow_html=True)

    df = load_history()

    if df.empty:
        st.info("No prediction records yet. Head to Dashboard to run your first prediction.")
    else:
        section(f"All Records — {len(df)} entries")

        # Filters
        f1, f2, _ = st.columns([1, 1, 3])

        with f1:
            if "Crop" in df.columns:
                crop_filter = st.multiselect(
                    "Filter by Crop",
                    df["Crop"].unique(),
                    default=list(df["Crop"].unique())
                )
                df = df[df["Crop"].isin(crop_filter)]

        with f2:
            if "State" in df.columns:
                state_filter = st.multiselect(
                    "Filter by State",
                    df["State"].unique(),
                    default=list(df["State"].unique())
                )
                df = df[df["State"].isin(state_filter)]

        # ✅ FIXED INDENTATION (yahi main issue tha)
        df["Yield"] = df["Yield"].round(2)
        df["Temp"] = df["Temp"].round(1)

        st.dataframe(df.reset_index(drop=True), use_container_width=True)

        # Download button
        dl_col, _ = st.columns([1, 5])
        with dl_col:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Export CSV",
                csv,
                file_name=f"cropiq_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ============================================================
#  ABOUT
# ============================================================
elif "About" in menu:

    st.markdown("""
    <div class="about-hero animate">
        <div class="about-title">CropIQ</div>
        <div style="font-size:12px;text-transform:uppercase;letter-spacing:3px;color:var(--t3);margin-top:4px">Smart Farming Intelligence</div>
        <div class="about-desc">
            An AI-powered agricultural decision support system that combines real-time weather data
            with machine learning to predict crop yield and recommend the best crop for your land.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("Core Features")
    st.markdown("""
    <div class="feature-grid animate">
        <div class="feature-item">
            <div class="fi-icon">🤖</div>
            <div class="fi-title">RandomForest ML Model</div>
            <div class="fi-desc">Trained on real APY (Area–Production–Yield) datasets for Rice, Wheat, Arhar, and more with 300 estimators.</div>
        </div>
        <div class="feature-item">
            <div class="fi-icon">🌦</div>
            <div class="fi-title">Live Weather Integration</div>
            <div class="fi-desc">Fetches real-time temperature, humidity, and rainfall via OpenWeatherMap API for any city in India.</div>
        </div>
        <div class="feature-item">
            <div class="fi-icon">🌿</div>
            <div class="fi-title">5-Crop Database</div>
            <div class="fi-desc">Supports Rice, Wheat, Maize, Arhar, and Sugarcane with ideal condition ranges and market pricing.</div>
        </div>
        <div class="feature-item">
            <div class="fi-icon">📈</div>
            <div class="fi-title">Profit Forecasting</div>
            <div class="fi-desc">Calculates estimated gross income and net profit based on predicted yield and current market rates.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section("Model Architecture")
    st.markdown("""
    <div class="kpi animate">
        <table class="ctable">
            <tr><td>Algorithm</td><td>RandomForestRegressor (sklearn)</td></tr>
            <tr><td>Estimators</td><td>300 trees, max depth 12</td></tr>
            <tr><td>Features</td><td>Area, Temperature, Rainfall, Humidity</td></tr>
            <tr><td>Training Data</td><td>APY district-level agricultural datasets</td></tr>
            <tr><td>Target</td><td>Yield (Production ÷ Area)</td></tr>
            <tr><td>Persistence</td><td>Serialized via joblib → model.pkl</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)