import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
import base64

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model artifacts (bundled in repo, no download needed) ──
from tensorflow.keras.models import load_model

model           = load_model("heart_disease_model.keras")
scaler          = joblib.load("heart_scaler.pkl")
feature_columns = joblib.load("heart_features.pkl")

feature_mapping = {
    "age"                      : "Patient Age",
    "trestbps"                 : "Resting Blood Pressure",
    "chol"                     : "Cholesterol Level",
    "thalch"                   : "Maximum Heart Rate",
    "oldpeak"                  : "ST Depression",
    "ca"                       : "Major Vessels Count",
    "fbs_True"                 : "High Fasting Blood Sugar",
    "fbs_False"                : "Normal Fasting Blood Sugar",
    "cp_typical angina"        : "Typical Angina Chest Pain",
    "cp_atypical angina"       : "Atypical Angina Chest Pain",
    "cp_non-anginal"           : "Non-Anginal Chest Pain",
    "cp_asymptomatic"          : "Asymptomatic Chest Pain",
    "slope_upsloping"          : "Upsloping ST Segment",
    "slope_flat"               : "Flat ST Segment",
    "slope_downsloping"        : "Downsloping ST Segment",
    "sex_Female"               : "Female Gender",
    "sex_Male"                 : "Male Gender",
    "restecg_normal"           : "Normal ECG",
    "restecg_st-t abnormality" : "ST-T Wave Abnormality ECG",
    "restecg_lv hypertrophy"   : "Left Ventricular Hypertrophy ECG",
    "thal_normal"              : "Normal Thalassemia",
    "thal_fixed defect"        : "Fixed Thalassemia Defect",
    "thal_reversable defect"   : "Reversible Thalassemia Defect",
    "exang_True"               : "Exercise Induced Angina",
    "exang_False"              : "No Exercise Induced Angina",
    "dataset_Cleveland"        : "Cleveland Dataset",
    "dataset_Hungary"          : "Hungary Dataset",
    "dataset_Switzerland"      : "Switzerland Dataset",
    "dataset_VA Long Beach"    : "VA Long Beach Dataset",
}

recommendations = {
    "Exercise Induced Angina"         : "Avoid strenuous physical activity until evaluated. Consult a cardiologist for a stress test. Monitor active heart rate regularly.",
    "Asymptomatic Chest Pain"         : "Silent chest pain is a serious indicator. Schedule an immediate cardiac evaluation.",
    "Typical Angina Chest Pain"       : "Typical angina requires medical review. Avoid triggers like cold weather and heavy meals.",
    "High Fasting Blood Sugar"        : "Manage blood sugar through diet and medication. Consult an endocrinologist.",
    "Cholesterol Level"               : "Reduce saturated fats and increase fibre intake. Schedule a lipid profile test.",
    "ST Depression"                   : "ST depression indicates reduced blood flow. Seek immediate cardiac evaluation.",
    "Downsloping ST Segment"          : "Downsloping ST segment is a high-risk ECG pattern. Consult a cardiologist urgently.",
    "Flat ST Segment"                 : "Flat ST segment may indicate ischaemia. Schedule an ECG review.",
    "Fixed Thalassemia Defect"        : "Fixed thalassemia defect indicates permanent heart damage. Regular cardiac monitoring required.",
    "Reversible Thalassemia Defect"   : "Reversible defect suggests reduced blood flow. Nuclear stress test recommended.",
    "Major Vessels Count"             : "Blocked vessels significantly increase risk. Angiography evaluation recommended.",
    "Resting Blood Pressure"          : "High blood pressure strains the heart. Reduce salt intake and monitor daily.",
    "Patient Age"                     : "Age is a non-modifiable risk factor. Increase frequency of cardiac checkups.",
    "Left Ventricular Hypertrophy ECG": "LV hypertrophy indicates heart strain. Blood pressure management is critical.",
}


def get_recommendation(feature_name):
    for key in recommendations:
        if key.lower() in feature_name.lower():
            return recommendations[key]
    return "Maintain a healthy lifestyle. Regular exercise, balanced diet, and routine checkups are recommended."


def compute_feature_importance(patient_scaled):
    """Fast perturbation-based feature importance (no SHAP — avoids worker timeouts)."""
    base_pred = float(model.predict(patient_scaled, verbose=0)[0][0])
    importances = []
    for i in range(patient_scaled.shape[1]):
        perturbed = patient_scaled.copy()
        perturbed[0, i] = 0.0
        pred = float(model.predict(perturbed, verbose=0)[0][0])
        importances.append(pred - base_pred)
    return np.array(importances)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        patient = pd.DataFrame([{
            "age"     : float(data["age"]),
            "sex"     : data["sex"],
            "dataset" : data["dataset"],
            "cp"      : data["cp"],
            "trestbps": float(data["trestbps"]),
            "chol"    : float(data["chol"]),
            "fbs"     : data["fbs"],
            "restecg" : data["restecg"],
            "thalch"  : float(data["thalch"]),
            "exang"   : data["exang"],
            "oldpeak" : float(data["oldpeak"]),
            "slope"   : data["slope"],
            "ca"      : float(data["ca"]),
            "thal"    : data["thal"],
        }])

        patient_encoded = pd.get_dummies(patient)
        patient_encoded = patient_encoded.reindex(columns=feature_columns, fill_value=0)
        patient_scaled  = scaler.transform(patient_encoded)

        probability = float(model.predict(patient_scaled, verbose=0)[0][0])

        if probability < 0.40:
            risk_class  = "low"
            result_text = "✓ LOW RISK OF HEART DISEASE"
        elif probability < 0.60:
            risk_class  = "moderate"
            result_text = "⚠ MODERATE RISK OF HEART DISEASE"
        else:
            risk_class  = "high"
            result_text = "🔴 VERY HIGH RISK OF HEART DISEASE"

        confidence = round(max(probability, 1 - probability) * 100, 2)

        # Feature importance (fast perturbation method)
        vals      = compute_feature_importance(patient_scaled)
        top_idx   = np.argsort(np.abs(vals))[-10:]
        top_vals  = vals[top_idx]
        top_names = [feature_mapping.get(feature_columns[i], feature_columns[i]) for i in top_idx]
        colors    = ["#f85149" if v > 0 else "#3fb950" for v in top_vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_names, top_vals, color=colors)
        ax.axvline(0, color="white", linewidth=0.8)
        ax.set_title("Top 10 Features Influencing This Prediction", color="#e6edf3", fontsize=12, pad=10)
        ax.set_xlabel("Importance  (🔴 increases risk · 🟢 reduces risk)", color="#8a9bb0", fontsize=9)
        ax.tick_params(colors="#8a9bb0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

        top_risk_idx  = int(np.argmax(top_vals)) if top_vals.max() > 0 else None
        top_prot_idx  = int(np.argmin(top_vals)) if top_vals.min() < 0 else None
        top_risk_name = top_names[top_risk_idx] if top_risk_idx is not None else "N/A"
        top_prot_name = top_names[top_prot_idx] if top_prot_idx is not None else "N/A"
        recommendation = get_recommendation(top_risk_name)

        return jsonify({
            "probability"   : round(probability, 4),
            "confidence"    : confidence,
            "risk_class"    : risk_class,
            "result_text"   : result_text,
            "shap_chart"    : chart_b64,
            "top_risk"      : top_risk_name,
            "top_protect"   : top_prot_name,
            "recommendation": recommendation,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
