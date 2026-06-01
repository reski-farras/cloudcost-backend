import joblib

model = joblib.load("model_prediksi_biaya.pkl")
booster = model.get_booster()

for imp_type in ["weight", "gain", "cover"]:
    score = booster.get_score(importance_type=imp_type)
    print(f"\nImportance Type: {imp_type}")
    # Sort by importance value
    sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
    for feat, val in sorted_score:
        print(f"  {feat}: {val:.4f}")
