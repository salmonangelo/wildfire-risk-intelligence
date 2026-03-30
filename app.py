from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)

# -----------------------------
# Load Model (Lazy Loading)
# -----------------------------
MODEL_PATH = os.path.join("model", "wildfire_rf_model.pkl")
model = None

def get_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

# -----------------------------
# Safe Float Conversion
# -----------------------------
def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except:
        return default

# -----------------------------
# Features (must match training)
# -----------------------------
FEATURES = [
    "temp_mean",
    "temp_range",
    "humidity_min",
    "wind_speed_max",
    "pressure_mean",
    "solar_radiation_mean",
    "cloud_cover_mean"
]

# -----------------------------
# Cities
# -----------------------------
CITIES = {
    "chennai": {"lat": 13.0827, "lon": 80.2707},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "los_angeles": {"lat": 34.0522, "lon": -118.2437},
    "sydney": {"lat": -33.8688, "lon": 151.2093},
    "lisbon": {"lat": 38.7223, "lon": -9.1393}
}

# -----------------------------
# Risk Mapping
# -----------------------------
def get_risk_level(probability: float) -> str:
    if probability < 0.33:
        return "LOW"
    elif probability < 0.66:
        return "MEDIUM"
    else:
        return "HIGH"

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Weather API
# -----------------------------
@app.route("/weather-by-city", methods=["POST"])
def weather_by_city():
    try:
        city = request.json.get("city")

        if city not in CITIES:
            return jsonify({"error": "Invalid city selected"}), 400

        lat = CITIES[city]["lat"]
        lon = CITIES[city]["lon"]

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,wind_speed_10m,cloud_cover"
            "&hourly=relative_humidity_2m,surface_pressure,shortwave_radiation"
        )

        response = requests.get(url, timeout=10).json()

        weather = {
            "temp_mean": safe_float(response["current"].get("temperature_2m")),
            
            # convert m/s → km/h
            "wind_speed_max": safe_float(response["current"].get("wind_speed_10m")) * 3.6,
            
            "cloud_cover_mean": safe_float(response["current"].get("cloud_cover")),
            "pressure_mean": safe_float(response["hourly"]["surface_pressure"][0]),
            "humidity_min": safe_float(response["hourly"]["relative_humidity_2m"][0]),
            "solar_radiation_mean": safe_float(response["hourly"]["shortwave_radiation"][0]),
            "temp_range": 5
        }

        return jsonify(weather)

    except Exception as e:
        print("WEATHER ERROR:", str(e))
        return jsonify({
            "error": "Weather data fetch failed",
            "details": str(e)
        }), 500

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Safe input handling
        input_row = {f: safe_float(data.get(f)) for f in FEATURES}

        # Ensure correct order
        X = pd.DataFrame([input_row])[FEATURES]

        model = get_model()

        probability = float(model.predict_proba(X)[0][1])
        risk_level = get_risk_level(probability)

        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        top_factors = importance_df.head(3)["feature"].tolist()

        return jsonify({
            "probability": round(probability, 2),
            "risk_level": risk_level,
            "top_factors": top_factors
        })

    except Exception as e:
        print("PREDICT ERROR:", str(e))   # 👈 VERY IMPORTANT
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)