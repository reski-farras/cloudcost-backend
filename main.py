from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import joblib
import numpy as np
import pandas as pd
import warnings

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
        cpu_efficiency = (data.Actual_CPU_Hours / data.Required_CPU_Hours) if data.Required_CPU_Hours > 0 else 1.0

        df = pd.DataFrame([{
            "Storage_Used_GB": float(data.Storage_Used_GB),
            "Required_CPU_Hours": float(data.Required_CPU_Hours),
            "Actual_CPU_Hours": float(data.Actual_CPU_Hours),
            "CPU_Utilization_%": float(data.CPU_Utilization),
            "Region": data.Region,
            "Billing_Period": data.Billing_Period,
            "Service_Category": data.Service_Category,
            "Instance_Status": data.Instance_Status,
            "CPU_Efficiency": float(cpu_efficiency),
        }])

        categorical_cols = ["Region", "Billing_Period", "Service_Category", "Instance_Status"]
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                val = df[col].iloc[0]
                df[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
            else:
                df[col] = 0

        df = df.astype(float)
        hasil = model.predict(df.values)[0]
        hasil = float(np.clip(hasil, 3.70, 66.33))

        # Analisis tambahan
        potensi_penghematan = 0.0
        if data.Actual_CPU_Hours > data.Required_CPU_Hours and data.Required_CPU_Hours > 0:
            biaya_per_jam = data.Compute_Cost / data.Actual_CPU_Hours if data.Actual_CPU_Hours > 0 else 0
            potensi_penghematan = (data.Actual_CPU_Hours - data.Required_CPU_Hours) * biaya_per_jam
        if data.CPU_Utilization < 35:
            potensi_penghematan += data.Compute_Cost * 0.3
        potensi_penghematan = round(float(np.clip(potensi_penghematan, 0, hasil * 0.5)), 2)

        proyeksi_estimasi_biaya = round(float(hasil * cpu_efficiency), 2)

        if data.CPU_Utilization > 85:
            status_beban = "Kelebihan Beban"
        elif data.CPU_Utilization < 35:
            status_beban = "Kurang Dimanfaatkan"
        else:
            status_beban = "Optimal"

        ada_anomali = 0
        if data.CPU_Utilization > 95:
            ada_anomali = 1
        elif data.Network_Cost > (data.Compute_Cost * 1.5) and data.Compute_Cost > 0:
            ada_anomali = 1
        elif data.Actual_CPU_Hours > (data.Required_CPU_Hours * 2.0) and data.Required_CPU_Hours > 0:
            ada_anomali = 1
        elif data.Storage_Cost > (data.Storage_Used_GB * 2.0) and data.Storage_Used_GB > 0:
            ada_anomali = 1

        rekomendasi = "Pertahankan arsitektur cloud Anda yang efisien."
        if data.Network_Cost > (data.Compute_Cost * 1.5) and data.Compute_Cost > 0:
            rekomendasi = "Cek anomali pada biaya jaringan."
        elif data.CPU_Utilization < 35:
            rekomendasi = "Kurangi alokasi CPU karena utilisasi rendah."
        elif data.CPU_Utilization > 85:
            rekomendasi = "Tingkatkan kapasitas CPU untuk menghindari kegagalan sistem."
        elif data.Actual_CPU_Hours > data.Required_CPU_Hours:
            rekomendasi = "Sesuaikan jam CPU aktual agar sama dengan jam required."

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