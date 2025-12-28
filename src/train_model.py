import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
model_path = os.path.join(project_dir, "model", "wildfire_rf_model.pkl")

# 1. Load dataset
df = pd.read_csv(os.path.join(project_dir, "data", "final_dataset.csv"))

# 2. Separate features and target
X = df.drop("occured", axis=1)
y = df["occured"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Initialize Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Predict on test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluation
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# 9. Save trained model
joblib.dump(model, model_path)
print("\nModel saved successfully.")
