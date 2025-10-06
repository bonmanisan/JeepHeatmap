from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import linear
import os
from datetime import datetime

app = Flask(__name__)

# ============================================================
# 1️⃣ Load model and dataset
# ============================================================
MODEL_FILE = "jeep_pipeline.joblib"
DATA_FILE = "expandedDataset_with_JeepVolume.csv"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ jeep_pipeline.joblib not found — please train and export model first.")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("❌ expandedDataset_with_JeepVolume.csv not found — please include your dataset.")

artifact = joblib.load(MODEL_FILE)
rf_model = artifact["model"]
feature_cols = artifact["feature_cols"]

df = pd.read_csv(DATA_FILE)
df.columns = [c.lower().strip() for c in df.columns]

# Detect columns
lat_col = next((c for c in df.columns if "lat" in c), None)
lon_col = next((c for c in df.columns if "lon" in c), None)

if lat_col is None or lon_col is None:
    raise ValueError("Dataset must include latitude and longitude columns.")

df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})

print(f"✅ Loaded dataset: {len(df)} rows")

# ============================================================
# 2️⃣ Helper — auto detect current time & event from CSV
# ============================================================
def auto_fill_from_csv():
    now = datetime.now()
    current_day = now.strftime("%A")
    current_hour = now.hour
    month = now.month
    season = "Dry" if month in [12, 1, 2, 3, 4, 5] else "Wet"

    # Look up event from CSV for this day/hour
    event = "None"
    if "event" in df.columns:
        match = df[
            (df["dayofweek"].str.lower() == current_day.lower()) &
            (df["hour"] == current_hour)
        ]
        if not match.empty:
            event = match["event"].iloc[0]

    return {
        "dayofweek": current_day,
        "hour": current_hour,
        "season": season,
        "event": event
    }

# ============================================================
# 3️⃣ Predict API
# ============================================================
@app.route("/api/predict", methods=["GET"])
def predict():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Provide valid ?lat= and ?lon= query parameters"}), 400

    stop = request.args.get("stop", "")
    auto_data = auto_fill_from_csv()

    input_dict = {
        "latitude": lat,
        "longitude": lon,
        "stop": stop,
        "dayofweek": auto_data["dayofweek"],
        "hour": auto_data["hour"],
        "season": auto_data["season"],
        "event": auto_data["event"],
        "jeepvolume": 0
    }

    feat_df = pd.DataFrame([input_dict], columns=feature_cols)
    pred = float(rf_model.predict(feat_df)[0])

    return jsonify({
        **input_dict,
        "predicted_volume": round(pred, 2)
    })

# ============================================================
# 4️⃣ Dynamic Heatmap
# ============================================================
@app.route("/map")
def show_map():
    auto_data = auto_fill_from_csv()
    print(f"🌤 Using {auto_data}")

    # Prepare input for all stops in the dataset
    input_data = []
    for _, row in df.iterrows():
        input_data.append({
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "stop": row.get("stop", ""),
            "dayofweek": auto_data["dayofweek"],
            "hour": auto_data["hour"],
            "season": auto_data["season"],
            "event": auto_data["event"],
            "jeepvolume": 0
        })

    pred_df = pd.DataFrame(input_data, columns=feature_cols)
    df["predicted_volume"] = rf_model.predict(pred_df)

    # Normalize for heat intensity
    min_vol = df["predicted_volume"].min()
    max_vol = df["predicted_volume"].max()
    df["volume_norm"] = (df["predicted_volume"] - min_vol) / (max_vol - min_vol)

    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13)

    heat_data = df[["latitude", "longitude", "volume_norm"]].values.tolist()
    HeatMap(
        heat_data,
        radius=12,
        blur=15,
        max_zoom=13,
        gradient={0.0: "green", 0.5: "yellow", 1.0: "red"}
    ).add_to(m)

    # Color legend
    colormap = linear.YlOrRd_09.scale(min_vol, max_vol)
    colormap.caption = f"Predicted Jeep Volume ({auto_data['dayofweek']} {auto_data['hour']}:00, {auto_data['season']}, {auto_data['event']})"
    colormap.add_to(m)

    return render_template("map.html", map_html=m._repr_html_())

# ============================================================
# 5️⃣ Run app
# ============================================================
if __name__ == "__main__":
    print("🚀 Flask app running at http://127.0.0.1:5000")
    app.run(debug=True)
