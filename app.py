from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import folium
from folium.plugins import HeatMap
from branca.colormap import linear
import os

# ============================================================
# 1️⃣ Initialize Flask
# ============================================================
app = Flask(__name__)

# ============================================================
# 2️⃣ Load model and dataset
# ============================================================
MODEL_FILE = "jeep_pipeline.joblib"
DATA_FILE = "expandedDataset_with_JeepVolume.csv"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ jeep_pipeline.joblib not found — please train and export model first.")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("❌ expandedDataset_with_JeepVolume.csv not found — please include your dataset.")

# Load model + metadata
artifact = joblib.load(MODEL_FILE)
rf_model = artifact["model"]
feature_cols = artifact["feature_cols"]

# Load dataset
df = pd.read_csv(DATA_FILE)
df.columns = [c.lower() for c in df.columns]

# Detect columns
lat_col = next((c for c in df.columns if "lat" in c), None)
lon_col = next((c for c in df.columns if "lon" in c), None)
vol_col = next((c for c in df.columns if "volume" in c or "count" in c), None)

if lat_col is None or lon_col is None:
    raise ValueError("Dataset must include latitude and longitude columns.")

# If no volume column, create one by counting points
if vol_col is None:
    df["volume"] = df.groupby([lat_col, lon_col])[lat_col].transform("count")
else:
    df = df.rename(columns={vol_col: "volume"})

df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})

print(f"✅ Loaded dataset: {len(df)} rows")

# ============================================================
# 3️⃣ Routes
# ============================================================

@app.route("/")
def index():
    return render_template("map.html")


@app.route("/api/predict", methods=["GET"])
def predict():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Provide valid ?lat= and ?lon= query parameters"}), 400

    stop = request.args.get("stop", "")
    dayofweek = request.args.get("dayofweek", "")
    hour = request.args.get("hour", "")
    season = request.args.get("season", "")
    event = request.args.get("event", "")
    jeepvolume = request.args.get("jeepvolume", "")

    input_dict = {
        "latitude": lat,
        "longitude": lon,
        "stop": stop,
        "dayofweek": dayofweek,
        "hour": hour,
        "season": season,
        "event": event,
        "jeepvolume": jeepvolume
    }

    feat_df = pd.DataFrame([input_dict], columns=feature_cols)
    pred = float(rf_model.predict(feat_df)[0])

    return jsonify({
        "latitude": lat,
        "longitude": lon,
        "stop": stop,
        "dayofweek": dayofweek,
        "hour": hour,
        "season": season,
        "event": event,
        "jeepvolume": jeepvolume,
        "predicted_volume": round(pred, 2)
    })


@app.route("/api/map")
def map_data():
    """Return heatmap data as JSON for mobile or web app."""
    data = df[["latitude", "longitude", "volume"]].to_dict(orient="records")
    return jsonify(data)


@app.route("/map")
def show_map():
    """Render Folium map with heatmap overlay based on jeep volume."""
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13)

    # Normalize volume for weight
    min_vol = df["volume"].min()
    max_vol = df["volume"].max()
    df["volume_norm"] = (df["volume"] - min_vol) / (max_vol - min_vol)

    # Prepare heatmap data: [lat, lon, normalized_volume]
    heat_data = df[["latitude", "longitude", "volume_norm"]].values.tolist()

    # Add HeatMap: colors reflect jeepvolume
    HeatMap(
        heat_data,
        radius=12,
        blur=15,
        max_zoom=13,
        gradient={0.0: 'green', 0.5: 'yellow', 1.0: 'red'}  # low → green, medium → yellow, high → red
    ).add_to(m)

    # Add legend for actual jeep volume
    colormap = linear.YlOrRd_09.scale(min_vol, max_vol)
    colormap.caption = "Jeep Volume"
    colormap.add_to(m)

    map_html = m._repr_html_()
    return render_template("map.html", map_html=map_html)


# ============================================================
# 4️⃣ Run app
# ============================================================
if __name__ == "__main__":
    print("🚀 Flask app running at http://127.0.0.1:5000")
    app.run(debug=True)
