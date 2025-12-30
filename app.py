from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# -----------------------------
# App Initialization
# -----------------------------
# Get the absolute path to the templates directory
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'frontend')
app = Flask(__name__, template_folder=template_dir)

# -----------------------------
# Load Trained Model
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


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts weather inputs and returns wildfire risk prediction.
    """
    try:
        data = request.get_json()

        # Validate input
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing features: {', '.join(missing)}"
            }), 400

        # Build input dataframe
        input_row = {feature: float(data[feature]) for feature in FEATURES}
        X = pd.DataFrame([input_row])

        # Predict probability
        probability = float(model.predict_proba(X)[0][1])
        risk_level = get_risk_level(probability)

        # Feature importance (Explainability)
        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        top_factors = importance_df.head(3)["feature"].tolist()

        # Response
        response = {
            "probability": round(probability, 2),
            "risk_level": risk_level,
            "top_factors": top_factors
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# -----------------------------
# App Runner
# -----------------------------
if __name__ == "__main__":
    print("\n--- Wildfire Risk Prediction ---")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )