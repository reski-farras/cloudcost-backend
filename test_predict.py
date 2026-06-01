import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

model = joblib.load("model_prediksi_biaya.pkl")
le = joblib.load("label_encoders.pkl")

# Simulate exact same flow as main.py predict endpoint
data = {
    "Storage_Used_GB": 333,
    "Required_CPU_Hours": 720,
    "Actual_CPU_Hours": 720,
    "CPU_Utilization": 75,
    "Region": "Asia-East1",
    "Billing_Period": "Daily",
    "Service_Category": "Network",
    "Instance_Status": "Pending",
    "Compute_Cost": 232,
    "Network_Cost": 121,
    "Storage_Cost": 532,
}

cpu_efficiency = data["Actual_CPU_Hours"] / data["Required_CPU_Hours"] if data["Required_CPU_Hours"] > 0 else 1.0

df = pd.DataFrame([{
    "Storage_Used_GB": data["Storage_Used_GB"],
    "Required_CPU_Hours": data["Required_CPU_Hours"],
    "Actual_CPU_Hours": data["Actual_CPU_Hours"],
    "CPU_Utilization_%": data["CPU_Utilization"],
    "Region": data["Region"],
    "Billing_Period": data["Billing_Period"],
    "Service_Category": data["Service_Category"],
    "Instance_Status": data["Instance_Status"],
    "CPU_Efficiency": cpu_efficiency,
}])

print("Before encoding:")
print(f"  shape: {df.shape}")
print(f"  columns: {list(df.columns)}")
print(f"  dtypes:\n{df.dtypes}")
print()

# Encode categorical columns
categorical_cols = ["Region", "Billing_Period", "Service_Category", "Instance_Status"]
for col in categorical_cols:
    if col in le:
        val = df[col].iloc[0]
        if val in le[col].classes_:
            df[col] = int(le[col].transform([val])[0])
        else:
            print(f"  WARNING: '{val}' not in {col} classes: {list(le[col].classes_)}")
            df[col] = 0
    else:
        df[col] = 0

df = df.astype(float)
print("\nAfter encoding:")
print(f"  shape: {df.shape}")
print(f"  columns: {list(df.columns)}")
print(f"  values: {df.values.tolist()}")

try:
    result = model.predict(df)
    print(f"\nPrediction result: {result[0]}")
except Exception as e:
    print(f"\nPrediction ERROR: {e}")
    # Try with numpy array directly
    print("Trying with numpy array...")
    result = model.predict(df.values)
    print(f"Prediction with .values: {result[0]}")
