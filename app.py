
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ==========================
# LOAD MODEL FILES
# ==========================

model = load_model("heart_disease_model.keras")

scaler = joblib.load(
    "heart_scaler.pkl"
)

feature_columns = joblib.load(
    "heart_features.pkl"
)

# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# PREDICTION API
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        patient = pd.DataFrame({

            "age":[int(data["age"])],
            "sex":[data["sex"]],
            "dataset":[data["dataset"]],
            "cp":[data["cp"]],
            "trestbps":[float(data["trestbps"])],
            "chol":[float(data["chol"])],
            "fbs":[data["fbs"]],
            "restecg":[data["restecg"]],
            "thalch":[float(data["thalch"])],
            "exang":[data["exang"]],
            "oldpeak":[float(data["oldpeak"])],
            "slope":[data["slope"]],
            "ca":[float(data["ca"])],
            "thal":[data["thal"]]

        })

        # ==========================
        # ENCODE
        # ==========================

        patient_encoded = pd.get_dummies(
            patient
        )

        patient_encoded = patient_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ==========================
        # SCALE
        # ==========================

        patient_scaled = scaler.transform(
            patient_encoded
        )

        # ==========================
        # PREDICT
        # ==========================

        probability = float(
            model.predict(
                patient_scaled,
                verbose=0
            )[0][0]
        )

        confidence = round(
            max(
                probability,
                1 - probability
            ) * 100,
            2
        )

        # ==========================
        # RISK LEVEL
        # ==========================

        if probability < 0.40:

            risk = "Low Risk"

        elif probability < 0.60:

            risk = "Moderate Risk"

        else:

            risk = "High Risk"

        return jsonify({

            "success": True,

            "risk": risk,

            "probability":
                round(probability, 4),

            "confidence":
                confidence

        })

    except Exception as e:

        return jsonify({

            "success": False,
            "error": str(e)

        })


# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

