import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

# ================= LOAD DATA =================
files = [
    "Data/APY_Arhar.xls",
    "Data/APY_Rice.xls",
    "Data/APY_Wheat.xls",
    "Data/crop_production.csv"
]

dfs = []

for f in files:
    if os.path.exists(f):
        try:
            df = pd.read_excel(f) if f.endswith(".xls") else pd.read_csv(f)
            dfs.append(df)
        except:
            pass

if not dfs:
    raise ValueError("❌ No dataset found")

data = pd.concat(dfs, ignore_index=True)

# ================= CLEAN =================
data.columns = [c.strip().lower() for c in data.columns]

# remove duplicate columns
data = data.loc[:, ~data.columns.duplicated()]

# rename
data.rename(columns={
    "state_name": "state",
    "district_name": "district"
}, inplace=True)

# ================= FIND COLS =================
area_col = [c for c in data.columns if "area" in c][0]
prod_col = [c for c in data.columns if "prod" in c][0]

# ================= CLEAN NUMERIC =================
data[area_col] = pd.to_numeric(data[area_col], errors="coerce")
data[prod_col] = pd.to_numeric(data[prod_col], errors="coerce")

data = data.dropna(subset=[area_col, prod_col])
data = data[(data[area_col] > 0) & (data[prod_col] > 0)]

# ================= TARGET =================
data["yield"] = data[prod_col] / data[area_col]

# remove outliers
data = data[data["yield"] < data["yield"].quantile(0.99)]

# ================= SPEED BOOST =================
if len(data) > 20000:
    data = data.sample(n=12000, random_state=42)

# ================= SAFE SERIES =================
def get_series(df, col):
    if col not in df.columns:
        return pd.Series(["unknown"] * len(df))
    
    s = df.loc[:, col]
    
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
        
    return s.astype(str)

state = get_series(data, "state")
crop = get_series(data, "crop")
season = get_series(data, "season")

# ================= ENCODING =================
data["state_enc"] = pd.factorize(state)[0]
data["crop_enc"] = pd.factorize(crop)[0]
data["season_enc"] = pd.factorize(season)[0]

# ================= WEATHER =================
np.random.seed(42)

data["temp"] = np.random.normal(28, 5, len(data)).clip(10, 45)
data["rain"] = np.random.normal(100, 50, len(data)).clip(0, 300)
data["humidity"] = np.random.normal(60, 15, len(data)).clip(10, 100)

# ================= FEATURES =================
data["log_area"] = np.log1p(data[area_col])

state_avg = data["yield"].groupby(state).mean()
crop_avg = data["yield"].groupby(crop).mean()

data["state_avg"] = state.map(state_avg)
data["crop_avg"] = crop.map(crop_avg)

# ================= FINAL X =================
X = pd.DataFrame({
    "area": data[area_col],
    "log_area": data["log_area"],
    "temp": data["temp"],
    "rain": data["rain"],
    "humidity": data["humidity"],
    "crop": data["crop_enc"],
    "state": data["state_enc"],
    "season": data["season_enc"],
    "crop_avg": data["crop_avg"],
    "state_avg": data["state_avg"]
})

# 🔥 NaN FIX
X = X.fillna(0)

# target
y = np.log1p(data["yield"])

# ================= TRAIN =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=3
)

model.fit(X_train, y_train)

# ================= EVAL =================
y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)

score = r2_score(y_true, y_pred)

# ================= SAVE =================
joblib.dump(model, "model.pkl")

print("✅ Model trained successfully")
print(f"📊 Samples: {len(data)}")
print(f"📈 R2 Score: {round(score, 3)}")