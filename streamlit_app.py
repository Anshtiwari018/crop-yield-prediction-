import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

st.title("🌾 Crop Yield Prediction Dashboard")

# Inputs
col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Crop", ["Wheat", "Rice", "Arhar"])
    area = st.number_input("Area (Hectare)", value=50)

with col2:
    district = st.text_input("District", "kota")
    month = st.selectbox("Month", [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ])

if st.button("🚀 Predict Yield"):

    # Dummy prediction (same जैसा तुम्हारा UI)
    yield_value = round(random.uniform(30, 120), 2)

    temp = 25
    humidity = 50
    rain = 0

    st.success("Prediction Done ✅")

    col1, col2, col3 = st.columns(3)

    col1.metric("Yield", f"{yield_value} Tonnes")
    col2.metric("Temperature", f"{temp}°C")
    col3.metric("Rain", f"{rain} mm")

    st.subheader("🌱 Recommendation")

    st.write("**Best Crop:** Wheat")
    st.write("Confidence: 55/100")

    st.write("Alternative Crops:")
    st.progress(0.45, text="Rice (45)")
    st.progress(0.35, text="Arhar (35)")

    # Chart
    st.subheader("📈 Monthly Yield Trend")

    months = ["Mar","Apr","May","Jun","Jul","Aug"]
    values = [90, 95, 100, 92, 101, 93]

    df = pd.DataFrame({"Month": months, "Yield": values})
    st.line_chart(df.set_index("Month"))