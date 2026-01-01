from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import requests
import os

# -----------------------------
# App Initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Trained ML Model
# -----------------------------
MODEL_PATH = os.path.join("model", "wildfire_rf_model.pkl")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Feature Order (MUST MATCH TRAINING)
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
# Predefined Cities (lat, lon)
# -----------------------------
CITIES = {
    "chennai": {"lat": 13.0827, "lon": 80.2707},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "los_angeles": {"lat": 34.0522, "lon": -118.2437},
    "sydney": {"lat": -33.8688, "lon": 151.2093},
    "lisbon": {"lat": 38.7223, "lon": -9.1393}
}

# -----------------------------
# Risk Mapping Logic
# -----------------------------
def get_risk_level(probability: float) -> str:
    if probability < 0.33:
        return "LOW"
    elif probability < 0.66:
        return "MEDIUM"
    else:
        return "HIGH"

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    """
    Serves the frontend dashboard.
    """
    return render_template("index.html")


# -----------------------------
# Weather Auto-Fetch by City
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
            "temp_mean": response["current"]["temperature_2m"],
            "wind_speed_max": response["current"]["wind_speed_10m"],
            "cloud_cover_mean": response["current"]["cloud_cover"],
            "pressure_mean": response["hourly"]["surface_pressure"][0],
            "humidity_min": response["hourly"]["relative_humidity_2m"][0],
            "solar_radiation_mean": response["hourly"]["shortwave_radiation"][0],
            "temp_range": 5  # adjustable by user
        }

        return jsonify(weather)

    except Exception as e:
        return jsonify({
            "error": "Weather data fetch failed",
            "details": str(e)
        }), 500


# -----------------------------
# Wildfire Risk Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate required features
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing features: {', '.join(missing)}"
            }), 400

        # Build input DataFrame
        input_row = {f: float(data[f]) for f in FEATURES}
        X = pd.DataFrame([input_row])

        # Predict wildfire occurrence probability
        probability = float(model.predict_proba(X)[0][1])
        risk_level = get_risk_level(probability)

        # Feature importance (Explainability)
        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        top_factors = importance_df.head(3)["feature"].tolist()

        # Response
        return jsonify({
            "probability": round(probability, 2),
            "risk_level": risk_level,
            "top_factors": top_factors
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# -----------------------------
# App Runner
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
