# debug_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data with proper header handling
data = pd.read_csv("forestfires.csv")

print("=== DATA DIAGNOSIS ===")
print("\n1. First 5 rows:")
print(data.head())

print("\n2. Column names:")
print(data.columns.tolist())

print("\n3. Data types:")
print(data.dtypes)

print("\n4. Basic stats:")
print(data[["temp", "RH", "wind", "rain", "area"]].describe())

print("\n5. Area value counts (top 10):")
print(data["area"].value_counts().head(10))

# Check if area has non-numeric values
print("\n6. Unique area values (first 20):")
print(data["area"].unique()[:20])