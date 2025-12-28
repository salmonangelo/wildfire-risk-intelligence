import joblib
import pandas as pd
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# 1. Load trained model
model_path = project_root / "model" / "wildfire_rf_model.pkl"
model = joblib.load(model_path)

# 2. Feature names (MUST match training order)
features = [
    "temp_mean",
    "temp_range",
    "humidity_min",
    "wind_speed_max",
    "pressure_mean",
    "solar_radiation_mean",
    "cloud_cover_mean"
]

# 3. Take user input
print("\nEnter current weather conditions:\n")

user_data = {}
for feature in features:
    value = float(input(f"{feature}: "))
    user_data[feature] = value

# 4. Convert input to DataFrame
input_df = pd.DataFrame([user_data])

# 5. Predict probability
probability = model.predict_proba(input_df)[0][1]

# 6. Risk logic
if probability < 0.33:
    risk_level = "LOW"
elif probability < 0.66:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

# 7. Feature importance (for explanation)
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

top_factors = importance_df.head(3)["Feature"].tolist()

# 8. Output results
print("\n--- Wildfire Risk Assessment ---")
print(f"Risk Probability : {probability:.2f}")
print(f"Risk Level       : {risk_level}")

print("\nKey contributing factors:")
for factor in top_factors:
    print(f"- {factor}")

# 9. Alert message
if risk_level == "HIGH":
    print("\n⚠️ ALERT: Area requires close surveillance and preparedness.")
elif risk_level == "MEDIUM":
    print("\n⚠️ CAUTION: Monitor conditions and stay prepared.")
else:
    print("\n✅ Conditions currently indicate low wildfire risk.")
