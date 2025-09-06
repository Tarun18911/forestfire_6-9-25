import pandas as pd
import numpy as np

def create_synthetic_dataset():
    """
    Creates a synthetic dataset to balance the number of rainy-day records.
    """
    try:
        data = pd.read_csv("forestfires.csv")
    except FileNotFoundError:
        print("Error: 'forestfires.csv' not found. Please make sure the file is in the same directory.")
        return

    data['fire_occurred'] = (data['area'] > 0).astype(int)

    low_risk_days = data[(data['fire_occurred'] == 0) & (data['rain'] == 0)]
    num_synthetic_records = 200
    synthetic_base = low_risk_days.sample(n=num_synthetic_records, replace=True, random_state=42)

    synthetic_data = synthetic_base.copy()
    synthetic_data['rain'] = np.random.uniform(0.1, 1.0, size=len(synthetic_data))
    synthetic_data['FFMC'] = synthetic_data['FFMC'] * np.random.uniform(0.8, 0.95, size=len(synthetic_data))
    synthetic_data['DMC'] = synthetic_data['DMC'] * np.random.uniform(0.7, 0.9, size=len(synthetic_data))
    synthetic_data['DC'] = synthetic_data['DC'] * np.random.uniform(0.6, 0.8, size=len(synthetic_data))
    synthetic_data['ISI'] = synthetic_data['ISI'] * np.random.uniform(0.5, 0.8, size=len(synthetic_data))

    synthetic_data['fire_occurred'] = 0
    synthetic_data['area'] = 0
    
    combined_data = pd.concat([
        data,
        synthetic_data
    ], ignore_index=True)

    # The 'fire_occurred' column is dropped because the training script will recreate it.
    combined_data.to_csv("forestfires_augmented.csv", index=False)

    print(f"✅ Corrected synthetic dataset created with {len(combined_data)} records.")
    print("The new dataset is saved as 'forestfires_augmented.csv'.")

if __name__ == "__main__":
    create_synthetic_dataset()