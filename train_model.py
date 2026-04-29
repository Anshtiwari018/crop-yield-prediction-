import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

# ================= LOAD DATA =================
# Adjust paths if your data is in a subfolder
XLS_FILES = [
    ("Data/APY_Arhar.xls", "Arhar/Tur"),
    ("Data/APY_Rice.xls", "Rice"),
    ("Data/APY_Wheat.xls", "Wheat"),
]
CSV_FILE = "Data/crop_production.csv"

dfs = []

# ── XLS files (crop-specific APY data) ──────────────
for path, crop_name in XLS_FILES:
    if not os.path.exists(path):
        print(f"⚠  Not found: {path}  (skipping)")
        continue
    try:
        df = pd.read_excel(path, engine="xlrd")
        df.columns = [c.strip().lower() for c in df.columns]
        # Rename common column variants
        df.rename(
            columns={
                "state_name": "state",
                "state name": "state",
                "district_name": "district",
                "district name": "district",
                "crop_year": "year",
                "crop year": "year",
            },
            inplace=True,
        )
        df["crop"] = crop_name  # inject crop name
        dfs.append(df)
        print(f"✅ Loaded {path}  →  {len(df)} rows")
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")

# ── Main CSV (multi-crop) ────────────────────────────
if os.path.exists(CSV_FILE):
    try:
        df_csv = pd.read_csv(CSV_FILE)
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
        df_csv.rename(
            columns={
                "state_name": "state",
                "district_name": "district",
                "crop_year": "year",
            },
            inplace=True,
        )
        dfs.append(df_csv)
        print(f"✅ Loaded {CSV_FILE}  →  {len(df_csv)} rows")
    except Exception as e:
        print(f"❌ Error reading {CSV_FILE}: {e}")
else:
    print(f"⚠  Not found: {CSV_FILE}  (skipping)")

if not dfs:
    raise ValueError("❌ No dataset found. Check your Data/ folder.")

data = pd.concat(dfs, ignore_index=True)
print(f"\n📦 Total rows before cleaning: {len(data)}")

# ================= CLEAN =================
data = data.loc[:, ~data.columns.duplicated()]

# Find area / production columns robustly
area_col = next((c for c in data.columns if "area" in c), None)
prod_col = next((c for c in data.columns if "produc" in c), None)

if area_col is None or prod_col is None:
    raise ValueError(
        f"❌ Cannot find area/production columns. Found: {data.columns.tolist()}"
    )

data[area_col] = pd.to_numeric(data[area_col], errors="coerce")
data[prod_col] = pd.to_numeric(data[prod_col], errors="coerce")
data.dropna(subset=[area_col, prod_col], inplace=True)
data = data[(data[area_col] > 0) & (data[prod_col] > 0)]

# ================= TARGET =================
data["yield"] = data[prod_col] / data[area_col]

# Remove top-1% outliers (coconut, sugarcane inflate badly)
data = data[data["yield"] < data["yield"].quantile(0.99)]
print(f"📦 Rows after cleaning: {len(data)}")

# ================= SAMPLE FOR SPEED =================
if len(data) > 20000:
    data = data.sample(n=15000, random_state=42)


# ================= SAFE SERIES =================
def safe_series(df, col):
    if col not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(str).str.strip()


state_s = safe_series(data, "state")
crop_s = safe_series(data, "crop")
season_s = safe_series(data, "season")

# ================= ENCODING =================
# Keep label-encoders so main.py can use the same mapping
state_enc, state_cats = pd.factorize(state_s)
crop_enc, crop_cats = pd.factorize(crop_s)
season_enc, season_cats = pd.factorize(season_s)

data["state_enc"] = state_enc
data["crop_enc"] = crop_enc
data["season_enc"] = season_enc

# ================= LOAD WEATHER =================
import difflib

# ================= LOAD WEATHER =================
weather = pd.read_csv("Data/popular_cities_weather.csv")
weather.columns = [c.lower().strip() for c in weather.columns]
# ================= LOAD SOIL =================
soil = pd.read_csv("Data/soil.csv")
soil.columns = [c.lower().strip() for c in soil.columns]
# rename columns
weather["temp"] = weather["tavg"]
weather["rain"] = weather["prcp"]

# clean
weather["city"] = weather["city"].str.lower().str.strip()
data["state"] = data["state"].str.lower().str.strip()

# ================= MANUAL MAPPING (HIGH ACCURACY) =================
CITY_TO_STATE = {
    "jaipur": "rajasthan",
    "jodhpur": "rajasthan",
    "udaipur": "rajasthan",
    "mumbai": "maharashtra",
    "pune": "maharashtra",
    "nagpur": "maharashtra",
    "lucknow": "uttar pradesh",
    "kanpur": "uttar pradesh",
    "agra": "uttar pradesh",
    "bhopal": "madhya pradesh",
    "indore": "madhya pradesh",
    "ahmedabad": "gujarat",
    "surat": "gujarat",
    "chennai": "tamil nadu",
    "coimbatore": "tamil nadu",
    "bangalore": "karnataka",
    "mysore": "karnataka",
    "kolkata": "west bengal",
    "hyderabad": "telangana",
    "patna": "bihar",
    "ranchi": "jharkhand",
    "bhubaneswar": "odisha",
    "guwahati": "assam",
    "delhi": "delhi",
}

# apply manual mapping
weather["state"] = weather["city"].map(CITY_TO_STATE)

# ================= AUTO MAPPING (SMART FALLBACK) =================
states_list = states_list = data["state"].str.lower().str.strip().dropna().unique()


def auto_map(city):
    match = difflib.get_close_matches(city, states_list, n=1, cutoff=0.8)
    return match[0] if match else "unknown"


# apply auto mapping only where manual failed
weather["state"] = weather.apply(
    lambda row: row["state"] if pd.notna(row["state"]) else auto_map(row["city"]),
    axis=1,
)

# ================= DATE → YEAR =================
weather["year"] = pd.to_datetime(weather["date"]).dt.year

# ================= AGGREGATE (BEST PRACTICE) =================
weather = weather.groupby(["state", "year"]).mean(numeric_only=True).reset_index()

# ================= HUMIDITY =================
weather["humidity"] = 60

# ================= MERGE =================
data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
weather["year"] = pd.to_numeric(weather["year"], errors="coerce").astype("Int64")

# 🔥 THIS LINE WAS MISSING
data = data.merge(weather, on=["state", "year"], how="left")
# ================= FINAL CLEAN =================
if "temp" not in data.columns:
    print("❌ temp missing after merge")
    data["temp"] = 25

if "rain" not in data.columns:
    data["rain"] = 100

if "humidity" not in data.columns:
    data["humidity"] = 60

# ✅ FIXED (no warning)
data["temp"] = data["temp"].fillna(data["temp"].median())
data["rain"] = data["rain"].fillna(data["rain"].median())
data["humidity"] = data["humidity"].fillna(60)

# ================= FEATURE ENGINEERING =================

data["log_area"] = np.log1p(data[area_col])

crop_avg_map = data["yield"].groupby(crop_s).mean()
state_avg_map = data["yield"].groupby(state_s).mean()

data["crop_avg"] = crop_s.map(crop_avg_map)
data["state_avg"] = state_s.map(state_avg_map)

# ================= FINAL FEATURE MATRIX =================
X = pd.DataFrame(
    {
        "area": data[area_col].values,
        "log_area": data["log_area"].values,
        "temp": data["temp"].values,
        "rain": data["rain"].values,
        "humidity": data["humidity"].values,
        "crop": data["crop_enc"].values,
        "state": data["state_enc"].values,
        "season": data["season_enc"].values,
        "crop_avg": data["crop_avg"].values,
        "state_avg": data["state_avg"].values,
    }
)

X = X.fillna(0)

y = np.log1p(data["yield"])
# ================= TRAIN =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# ================= EVAL =================
y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)
score = r2_score(y_true, y_pred)

# ================= SAVE MODEL + METADATA =================
# Save crop/state average maps so main.py can look them up
meta = {
    "crop_avg_map": crop_avg_map.to_dict(),
    "state_avg_map": state_avg_map.to_dict(),
    "crop_cats": list(crop_cats),
    "state_cats": list(state_cats),
    "season_cats": list(season_cats),
}

joblib.dump(model, "model.pkl")
joblib.dump(meta, "model_meta.pkl")

print("\n✅ Model trained and saved → model.pkl + model_meta.pkl")
print(f"📊 Training samples : {len(X_train)}")
print(f"📈 R² Score         : {round(score, 3)}")
print(f"🌾 Unique crops     : {len(crop_cats)}")
print(f"🗺  Unique states    : {len(state_cats)}")
