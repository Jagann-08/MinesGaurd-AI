import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove spaces from column names
data.columns = data.columns.str.strip()

# Features and target
X = data.drop("risk", axis=1)
y = data["risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print("       MINEGUARD AI MODEL")
print("===================================")
print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")

print("===================================")
print("model.pkl saved successfully!")
print("===================================")
