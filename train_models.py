# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Portugal dataset
data = pd.read_csv("forestfires_augmented.csv")

# Use ALL weather features
feature_columns = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI']
X = data[feature_columns]

# BETTER TARGET: Use actual fire occurrence (area > 0 = fire)
data['fire_occurred'] = (data['area'] > 0).astype(int)
y = data['fire_occurred']

print("Fire Occurrence distribution:")
print(y.value_counts())
print(f"Baseline accuracy: {max(y.value_counts()) / len(y):.2f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Fire Occurrence Prediction Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "forest_fire_model.pkl")
print("✅ Model saved!")