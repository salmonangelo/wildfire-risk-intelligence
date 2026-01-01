# 🔥 FireGuard: Explainable Wildfire Risk Intelligence

FireGuard is an **explainable machine learning–based decision-support system** that predicts wildfire occurrence risk using meteorological conditions.  
Instead of producing only a binary prediction, FireGuard provides a **risk probability, risk level, and key contributing environmental factors**, making the system transparent and actionable.

This project was built as part of a global hackathon challenge and is designed as an **offline, reproducible demo** with optional live weather integration.

---

## 🚨 Problem

Wildfires cause severe environmental damage, economic loss, and threats to human life.  
Identifying areas that require **careful surveillance** based on environmental conditions is difficult, especially at a local scale.

Most existing systems:
- Act as black boxes with no explanation, or  
- Focus only on prediction without decision support  

---

## 💡 Solution

FireGuard uses a **Random Forest machine learning model** trained on historical wildfire and meteorological data to estimate the **probability of wildfire occurrence**.

### Key capabilities:
- Predicts wildfire occurrence probability from weather data
- Converts probability into **Low / Medium / High** risk levels
- Explains *why* a risk level is assigned using feature importance
- Provides a clean, web-based dashboard for interaction
- Supports both **API-based auto weather fetch** and **manual override**

FireGuard is intended as a **decision-support tool**, not a real-time alert or prevention system.

---

## 🧠 System Architecture

Frontend (HTML/CSS Dashboard)
↓
Flask Backend API
↓
Random Forest ML Model
↓
Risk Probability + Risk Level
↓
Explainable Feature Importance


---

## 🧪 Model & Data

### Machine Learning Model
- **Algorithm:** Random Forest Classifier
- **Library:** Scikit-learn
- **Reason:** Robust to noise, handles non-linear relationships, and provides feature importance for explainability

### Dataset
- Public historical wildfire and meteorological data
- Derived from:
  - Satellite-based fire observations (NASA FIRMS / VIIRS)
  - Meteorological variables (Open-Meteo)
- Over **118,000 samples**
- Fully numerical, no missing values

### Input Features
- Mean temperature
- Temperature range
- Minimum humidity
- Maximum wind speed
- Atmospheric pressure
- Solar radiation
- Cloud cover

### Output
- Wildfire occurrence probability
- Risk level: **LOW / MEDIUM / HIGH**
- Top contributing meteorological factors

---

## 🌐 Web Dashboard

The dashboard allows users to:
- Select a predefined city (with known latitude & longitude)
- Auto-fetch live weather data using a public API
- Manually adjust parameters for scenario analysis
- Run ML-based wildfire risk assessment
- View explainable results and alert messaging

---

## 📸 Screenshots

