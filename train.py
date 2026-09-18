import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_excel("dataset.xlsx")

print("=" * 60)
print("MineGuard AI - Rockfall Prediction Model")
print("=" * 60)

print("\nDataset Shape :", data.shape)

print("\nClass Distribution")
print(data["risk"].value_counts())
X = data[[
    "slope_angle",
    "rainfall",
    "temperature",
    "vibration",
    "crack_width",
    "soil_moisture",
    "displacement",
    "pore_pressure"
]]

y = data["risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")
print(importance)
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")
print("=" * 60)