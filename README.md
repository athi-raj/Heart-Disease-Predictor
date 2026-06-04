<div align="center">

# 🫀 Heart Disease Predictor
### Explainable AI-Based Heart Risk Screening with Personalized Health Recommendations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a0p7yv38avhVELzzlh-pPjssC_KYlpAI?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Neural%20Network-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-FF6B6B)
![SMOTE](https://img.shields.io/badge/SMOTE-Class%20Balance-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**Analyze,Predict. Explain.**


A production-ready deep learning pipeline for clinical heart disease risk assessment with **Explainable AI (SHAP)** integration. Built on the UCI Heart Disease dataset, this project covers the full ML lifecycle — data ingestion, EDA, preprocessing, model training, evaluation, and an interactive GUI deployment — all within Google Colab.

</div>

---

## 🚨 Problem Statement

Cardiovascular disease is the **leading cause of death globally**, claiming over **17.9 million lives per year** (WHO, 2023). The majority are preventable with early detection — yet early diagnosis remains inaccessible to millions:

- Heart disease symptoms are **silent or misattributed** to unrelated conditions
- Manual clinical assessment is **time-consuming, costly, and error-prone**
- Patients in **rural and under-resourced communities** lack access to cardiologists
- Existing AI tools produce a result but **cannot explain their reasoning** — making them clinically unusable
- Most prediction models offer **no actionable guidance** after the result

> **The gap is not just prediction accuracy — it is explainability, accessibility, and actionable clinical guidance.**

---

## 💡 Solution

An **Explainable AI-powered heart risk screening system** that goes beyond a binary yes/no:

| Stage | What It Does |
|---|---|
| 🧩 **Analyze** | Processes and analyzes patient clinical and cardiovascular data |
| 🔍 **Predict** | Deep learning model classifies risk as Low / Moderate /  High |
| 🧠 **Explain** | SHAP identifies the exact clinical factors driving each individual prediction |
| 📊 **Visualise** | Dark-themed SHAP bar chart with human-readable feature labels per prediction |



---

## ✅ What Makes This Stand Out

| Capability | Status |
|---|---|
| Deep Learning Neural Network (TensorFlow / Keras) | ✅ |
| Explainable AI — SHAP per-patient feature contributions | ✅ |
| 3-Tier Risk Classification (Low / Moderate / Very High) | ✅ |
| Dynamic Threshold Optimisation (0.30–0.70 sweep) | ✅ |
| SMOTE Class Balancing on train set | ✅ |
| Human-Readable Feature Name Mapping (24 features) | ✅ |
| Top Risk Contributor identified per prediction | ✅ |
| Top Protective Contributor identified per prediction | ✅ |
| Confidence Score per Prediction | ✅ |
| Interactive Clinical GUI (sliders + dropdowns) | ✅ |
| Dark Theme UI | ✅ |
| Loading Indicator during SHAP computation | ✅ |
| Clinical Disclaimer on every prediction | ✅ |
| Zero Drive dependency — loads from GitHub repo | ✅ |

---

## 🎯 Impact

| Metric | Detail |
|---|---|
| **Who benefits** | Patients, GPs, rural health workers, preventive care programs |
| **Scale of problem** | 17.9 million cardiovascular deaths annually — majority preventable |
| **What is reduced** | Delayed diagnosis, unnecessary specialist referrals, late-stage treatment costs |
| **Key differentiator** | Prediction + Explainability + Recommendations — usable by non-specialists |

---

## 📊 Sample Output

```
Risk Level     : ⚠ MODERATE RISK OF HEART DISEASE
Confidence     : 72.43%
Risk Probability: 0.6218

🧠 AI Explanation
▲ Top Risk Contributor    : Exercise Induced Angina
▼ Top Protective Factor   : Normal Thalassemia


```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT LAYER                        │
│         14 Clinical Features → get_dummies           │
│              35 encoded features                     │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│               PREPROCESSING                          │
│   Median Imputation → SMOTE → StandardScaler         │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│             NEURAL NETWORK                           │
│  Dense(128,ReLU) → BatchNorm → Dropout(0.3)          │
│  Dense(64, ReLU) → BatchNorm → Dropout(0.2)          │
│  Dense(32, ReLU) → Dense(1, Sigmoid)                 │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│           DYNAMIC THRESHOLD (0.30–0.70)              │
│     Low Risk / Moderate Risk / Very High Risk        │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              SHAP EXPLAINER                          │
│   Top 10 feature contributions per patient           │
│              Top Risk Contributor                    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│         PERSONALIZED RECOMMENDATIONS                 │
│   Mapped from top SHAP risk contributor              │
│   to clinical action items                          │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Heart-Disease-Predictor/
├── heart_disease_uci.csv           # Raw UCI dataset (920 records, 15 features)
├── heart_disease_model.keras       # Trained Keras model (native format)
├── heart_scaler.pkl                # Fitted StandardScaler
├── heart_features.pkl              # Feature column schema for inference
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Excludes cache, checkpoints, OS files
└── Heart_Disease_Predictor.ipynb   # Main notebook — 5 cells
    ├── Cell 1 — Imports, Setup & Git Clone
    ├── Cell 2 — Pipeline Functions (EDA, Preprocessing, Model, Train, Evaluate, Save)
    ├── Cell 3 — Training, Evaluation & Save Functions
    ├── Cell 4 — Main Pipeline Execution
    └── Cell 5 — Interactive GUI + SHAP + Recommendations
```

---

## 🔍 Pipeline Walkthrough

### Cell 1 — Imports, Setup & Git Clone
Full dependency stack imported cleanly (no duplicates). Repo cloned directly from GitHub into Colab — dataset loads from `/content/Heart-Disease-Predictor/`, no Google Drive dependency. TF version and GPU confirmed.

### Cell 2 — Pipeline Functions
All stages as modular reusable functions:
- `load_data()` — CSV load, shape and column printout
- `perform_eda()` — missing values, target distribution, histogram grid, masked correlation heatmap
- `preprocess_data()` — median imputation, one-hot encoding, stratified 80/20 split, SMOTE on train only, StandardScaler fit on resampled set
- `build_model()` — `Dense(128→64→32→1)` with BatchNorm + Dropout, Adam + AUC
- `train_model()` — EarlyStopping (val_auc, patience=15) + ReduceLROnPlateau, 100 max epochs
- `evaluate_model()` — dynamic threshold sweep, Accuracy/Precision/Recall/F1/AUC, side-by-side Confusion Matrix + ROC Curve
- `save_model()` — persists all three artifacts to repo folder

### Cell 3 — Training, Evaluation & Save Functions
Full sklearn metrics suite imported. Dynamic threshold optimisation. Side-by-side Confusion Matrix and ROC Curve plots on a dark-styled figure.

### Cell 4 — Main Pipeline Execution
```
load_data → perform_eda → preprocess_data → build_model → train_model → evaluate_model → save_model
```

### Cell 5 — Interactive GUI + Explainable AI
- Loads model, scaler, feature schema from cloned repo
- SHAP `Explainer` built once at load time — fast per-prediction computation
- **14 input widgets** — sliders for numerics, dropdowns for categoricals
- **Loading indicator** shown during SHAP computation, cleared before result
- **3-tier risk card** — Low (green) / Moderate (amber) / Very High (red)
- **SHAP bar chart** — top 10 features, red = increases risk, green = reduces risk
- **24-feature human-readable mapping** — `cp_asymptomatic` → `Asymptomatic Chest Pain`
- **AI Explanation card** — Top Risk Contributor + Top Protective Contributor
- **Clinical disclaimer** — displayed on every prediction

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Records:** 920 patients — Cleveland, Hungary, Switzerland, VA Long Beach
- **Target:** `num` → binarised to `0` No Disease / `1` Disease Present
- **Class balance:** ~55% positive — corrected with SMOTE on train set only

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age in years |
| `sex` | Categorical | Male / Female |
| `cp` | Categorical | Chest pain type (4 types) |
| `trestbps` | Numeric | Resting blood pressure (mmHg) |
| `chol` | Numeric | Serum cholesterol (mg/dl) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | Resting ECG results (3 types) |
| `thalch` | Numeric | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina |
| `oldpeak` | Numeric | ST depression induced by exercise |
| `slope` | Categorical | Slope of ST segment (3 types) |
| `ca` | Numeric | Major vessels coloured by fluoroscopy (0–4) |
| `thal` | Categorical | Thalassemia type (3 types) |
| `dataset` | Categorical | Source institution |

---

## 📈 Model Architecture & Training

```
Input(35) → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
          → Dense(64,  ReLU) → BatchNorm → Dropout(0.2)
          → Dense(32,  ReLU)
          → Dense(1, Sigmoid)
```

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=0.0005) |
| Loss | Binary Crossentropy |
| Metrics | Accuracy, AUC |
| Early Stopping | monitor=val_auc, patience=15, restore best weights |
| LR Scheduler | ReduceLROnPlateau, factor=0.5, patience=7 |
| Max Epochs | 100 |
| Batch Size | 32 |
| Validation Split | 20% of train set |

---

## 🗂️ Saved Artifacts

| File | Description |
|---|---|
| `heart_disease_model.keras` | Trained Keras model — native format |
| `heart_scaler.pkl` | `StandardScaler` fit on SMOTE resampled train set |
| `heart_features.pkl` | Ordered feature list for inference schema alignment |

---

## 🔮 Future Scope

| Enhancement | Description |
|---|---|
| 🩺 **Wearable Integration** | Real-time risk monitoring via Apple Watch / Fitbit vitals |
| 📄 **PDF Report Generation** | Auto-generate clinical summary report per patient |
| 📱 **Mobile Deployment** | Flutter or React Native app with offline mode |
| 📡 **Continuous Monitoring** | Time-series risk tracking across multiple patient visits |


---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.20, Keras |
| Explainable AI | SHAP (PermutationExplainer) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Data & EDA | pandas, numpy, matplotlib, seaborn |
| ML Utilities | scikit-learn |
| Persistence | joblib |
| GUI | ipywidgets, IPython.display, HTML/CSS |
| Environment | Google Colab Free Tier |

---

## 🚀 Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h-ywzkRNOZDsp1NOk18sucCMg7qDIYsX?usp=sharing)

1. Click the badge above to open the notebook
2. Connect to a runtime — GPU recommended (`Runtime → Change runtime type → T4 GPU`)
3. `Runtime → Run all` (`Ctrl+F9`)
4. Use the **Cell 5 GUI** — enter patient details, click **Run Prediction**
5. Receive: Risk Level · Confidence Score · SHAP Chart · AI Explanation · Recommendations

---

## 🤝 Contributing

Pull requests are welcome. For major changes please open an issue first. Ensure updated cells follow the existing function-based sequential structure and all three artifacts are re-saved after retraining.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built for preventive healthcare · Powered by Explainable AI · Designed for accessibility**

*Prediction is not enough — explanation and action are what save lives.*

</div>
