import joblib
import pandas as pd
import warnings
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

# Encode categorical columns
categorical_cols = ["Region", "Billing_Period", "Service_Category", "Instance_Status"]
for col in categorical_cols:
    if col in le:
        val = df[col].iloc[0]
        if val in le[col].classes_:
            df[col] = int(le[col].transform([val])[0])
        else:
            df[col] = 0
    else:
        df[col] = 0

df = df.astype(float)

try:
    result = model.predict(df.values)[0]
except Exception as e:
    result = model.predict(df)[0]

import numpy as np
result = float(np.clip(result, 3.70, 66.33))
formatted_cost = f"${result:.2f}"

rekomendasi = "✅ Optimal"
if data["CPU_Utilization"] < 60:
    rekomendasi = "⚠️ Kurang Dimanfaatkan"
elif data["CPU_Utilization"] > 100:
    rekomendasi = "⚠️ Kelebihan Beban"

print("📊 HASIL ANALISIS CLOUD FINOPS")
print("==================================================")
print(f"💰 Estimasi Total Cost    : {formatted_cost}")
print(f"⚙️ Status Efisiensi CPU   : {cpu_efficiency:.2f}")
print(f"📌 Rekomendasi Sistem     : {rekomendasi}")

