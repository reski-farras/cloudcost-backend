from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import joblib
import numpy as np
import pandas as pd
import warnings
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model_prediksi_biaya.pkl")
label_encoders = joblib.load("label_encoders.pkl")

class BiayaInput(BaseModel):
    Project_Type: str
    Cloud_Service: str
    Service_Category: str
    Billing_Period: str
    Required_CPU_Hours: float
    Actual_CPU_Hours: float
    CPU_Utilization: float
    Storage_Used_GB: float
    Storage_Cost: float
    Compute_Cost: float
    Network_Cost: float
    Region: str
    Owner_Team: str
    Instance_Status: str
    Remarks: str

# Range realistis dari dataset
DATASET_STATS = {
    "Storage_Used_GB":    {"min": 1.0,   "max": 999.0,  "mean": 500.0},
    "Required_CPU_Hours": {"min": 1.0,   "max": 744.0,  "mean": 372.0},
    "Actual_CPU_Hours":   {"min": 1.0,   "max": 744.0,  "mean": 372.0},
    "CPU_Utilization_%":  {"min": 1.0,   "max": 99.0,   "mean": 50.0},
    "Total_Cost_$":       {"min": 3.70,  "max": 66.33,  "mean": 34.84},
}

@app.post("/predict")
def predict(data: BiayaInput):
    try:
        # Kalkulasi CPU Efficiency
        cpu_efficiency = (data.Actual_CPU_Hours / data.Required_CPU_Hours) if data.Required_CPU_Hours > 0 else 0.0

        # Susun data untuk XGBoost
        input_data = pd.DataFrame([{
            "Storage_Used_GB": float(data.Storage_Used_GB),
            "Required_CPU_Hours": float(data.Required_CPU_Hours),
            "Actual_CPU_Hours": float(data.Actual_CPU_Hours),
            "CPU_Utilization_%": float(data.CPU_Utilization),
            "Region": data.Region,
            "Billing_Period": data.Billing_Period,
            "Service_Category": data.Service_Category,
            "Instance_Status": data.Instance_Status,
            "CPU_Efficiency": float(cpu_efficiency)
        }])

        # Translasi Teks ke Angka (Label Encoder)
        for col in ["Region", "Billing_Period", "Service_Category", "Instance_Status"]:
            if col in label_encoders:
                le = label_encoders[col]
                val = input_data[col].iloc[0]
                input_data[col] = le.transform([val])[0] if val in le.classes_ else 0
            else:
                input_data[col] = 0

        # Tembak Model
        hasil = model.predict(input_data)[0]
        hasil = float(hasil)

        # Logika FinOps
        if cpu_efficiency < 0.6:
            status_beban = "Kurang Dimanfaatkan"
            rekomendasi = "⚠️ Underutilized (Kelebihan penyediaan)"
            potensi_penghematan = hasil * 0.35
        elif cpu_efficiency > 1.0:
            status_beban = "Kelebihan Beban"
            rekomendasi = "🔥 Overutilized (Perlu ditingkatkan skalanya)"
            potensi_penghematan = 0.0
        else:
            status_beban = "Optimal"
            rekomendasi = "✅ Dimanfaatkan sesuai kebutuhan secara efektif"
            potensi_penghematan = 0.0

        potensi_penghematan = round(float(potensi_penghematan), 2)
        proyeksi_estimasi_biaya = round(float(hasil * cpu_efficiency), 2)

        ada_anomali = 0
        if data.CPU_Utilization > 95:
            ada_anomali = 1
        elif data.Network_Cost > (data.Compute_Cost * 1.5) and data.Compute_Cost > 0:
            ada_anomali = 1
        elif data.Actual_CPU_Hours > (data.Required_CPU_Hours * 2.0) and data.Required_CPU_Hours > 0:
            ada_anomali = 1
        elif data.Storage_Cost > (data.Storage_Used_GB * 2.0) and data.Storage_Used_GB > 0:
            ada_anomali = 1

        # Akurasi dinamis
        base_accuracy = 90.19
        penalty = 0.0
        if data.CPU_Utilization > 95 or data.CPU_Utilization < 5:
            penalty += 8.0
        elif data.CPU_Utilization > 85 or data.CPU_Utilization < 15:
            penalty += 4.0
        if data.Required_CPU_Hours > 0:
            ratio = data.Actual_CPU_Hours / data.Required_CPU_Hours
            if ratio > 3.0 or ratio < 0.1:
                penalty += 7.0
            elif ratio > 2.0 or ratio < 0.3:
                penalty += 3.0
        if data.Compute_Cost > 0 and data.Network_Cost > (data.Compute_Cost * 2):
            penalty += 5.0
        if data.Storage_Used_GB > 900 and data.Storage_Cost < 1:
            penalty += 4.0
        akurasi_dinamis = round(max(base_accuracy - penalty, 60.0), 2)

        # Print summary to console
        print("\n📊 HASIL ANALISIS CLOUD FINOPS")
        print("==================================================")
        print(f"💰 Estimasi Total Cost    : ${round(hasil, 2):,.2f}")
        print(f"⚙️ Status Efisiensi CPU   : {cpu_efficiency:.2f}")
        print(f"📌 Rekomendasi Sistem     : {rekomendasi}\n")

        return {
            "prediksi_biaya": round(hasil, 2),
            "formatted": f"${round(hasil, 2):,.2f}",
            "akurasi_prediksi": akurasi_dinamis,
            "analisis_tambahan": {
                "proyeksi_estimasi_biaya": proyeksi_estimasi_biaya,
                "nilai_potensi_penghematan": potensi_penghematan,
                "status_beban_kerja": status_beban,
                "indikator_deteksi_anomali": ada_anomali,
                "rekomendasi_tindakan": rekomendasi
            },
            "input_parameters": {
                "Project_Type": data.Project_Type,
                "Cloud_Service": data.Cloud_Service,
                "Service_Category": data.Service_Category,
                "Billing_Period": data.Billing_Period,
                "Required_CPU_Hours": float(data.Required_CPU_Hours),
                "Actual_CPU_Hours": float(data.Actual_CPU_Hours),
                "CPU_Utilization": float(data.CPU_Utilization),
                "CPU_Efficiency": round(float(cpu_efficiency), 4),
                "Storage_Used_GB": float(data.Storage_Used_GB),
                "Storage_Cost": float(data.Storage_Cost),
                "Compute_Cost": float(data.Compute_Cost),
                "Network_Cost": float(data.Network_Cost),
                "Region": data.Region,
                "Owner_Team": data.Owner_Team,
                "Instance_Status": data.Instance_Status,
                "Remarks": data.Remarks
            }
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "Cloud Cost Predictor API aktif!"}

@app.get("/options")
def get_options():
    options = {}
    categorical_cols = ["Region", "Billing_Period", "Service_Category", "Instance_Status"]
    for col in categorical_cols:
        if col in label_encoders:
            options[col] = list(label_encoders[col].classes_)
    return options

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)