

# 🫀 Heart Disease Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a0p7yv38avhVELzzlh-pPjssC_KYlpAI?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Neural%20Network-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**A production-ready deep learning pipeline for clinical heart disease risk assessment.**
Built on the UCI Heart Disease dataset, this project covers the full ML lifecycle — from raw data ingestion and exploratory analysis through model training, evaluation, artifact persistence, and an interactive GUI deployment — all within Google Colab.



---

## 🎯 Objective

Cardiovascular disease is the leading cause of death globally. Early detection significantly improves patient outcomes. This project builds a binary classification model that takes 14 clinical features from a patient's medical record and outputs a **risk probability score**, enabling clinicians and researchers to flag high-risk individuals before symptoms escalate.

---


## 💡 Solution Approach

The problem is framed as binary classification on tabular clinical data. Rather than a traditional ML approach (Logistic Regression / Random Forest), a fully connected neural network with BatchNormalization is used to capture non-linear feature interactions. EarlyStopping on AUC (not accuracy) ensures the model optimises for ranking quality, which is more meaningful on mildly imbalanced medical data.


---

## 🧠 Technical Highlights

- **End-to-end ML pipeline** — data ingestion → EDA → preprocessing → training → evaluation → deployment
- **Neural network classifier** with BatchNormalization for training stability on small medical datasets
- **Stratified splitting** to preserve class balance across train/test sets
- **Early stopping on AUC** (not accuracy) — more meaningful on imbalanced clinical data
- **Artifact persistence** — model, scaler, and feature schema saved for reproducible inference
- **Interactive GUI** via `ipywidgets` — zero console input, production-style UX in a notebook

---



## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Records:** 920 patients across 4 clinical institutions (Cleveland, Hungary, Switzerland, VA Long Beach)
- **Target:** `num` column binarised → `0` No Disease / `1` Disease Present
- **Positive rate:** ~55% (mild imbalance — handled via stratified split and AUC-based monitoring)

| Feature | Type | Description | Example |
|---|---|---|---|
| `id` | Numeric | Patient ID | 1, 2, 3 |
| `age` | Numeric | Age in years | 63, 67 |
| `sex` | Categorical | Male / Female | Male |
| `dataset` | Categorical | Source institution | Cleveland, Hungary, Switzerland, VA Long Beach |
| `cp` | Categorical | Chest pain type | typical angina, asymptomatic |
| `trestbps` | Numeric | Resting blood pressure (mmHg) | 145, 160 |
| `chol` | Numeric | Serum cholesterol (mg/dl) | 233, 286 |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl | TRUE / FALSE |
| `restecg` | Categorical | Resting ECG results | lv hypertrophy, normal |
| `thalch` | Numeric | Maximum heart rate achieved | 150, 108 |
| `exang` | Categorical | Exercise-induced angina | TRUE / FALSE |
| `oldpeak` | Numeric | ST depression induced by exercise | 2.3, 1.5 |
| `slope` | Categorical | Slope of peak exercise ST segment | downsloping, flat |
| `ca` | Numeric | Major vessels coloured by fluoroscopy (0–4) | 0, 3 |
| `thal` | Categorical | Thalassemia type | fixed defect, normal, reversable defect |
| `num` | Target | Heart disease diagnosis | 0 = No Disease, 1+ = Disease |

---

## ⚙️ Pipeline Walkthrough

### Cell 1 — Setup, Imports & Load Data
Mounts Google Drive, imports the full dependency stack (pandas, numpy, matplotlib, seaborn, sklearn, TensorFlow/Keras, joblib), suppresses deprecation warnings, and loads the raw CSV. Prints dataset shape, column list, and confirms TensorFlow version and GPU availability on startup.

### Cell 2 — EDA & Preprocessing
Separates numeric and categorical columns automatically via `select_dtypes`. Applies **median imputation** to numeric nulls (preferred over mean for skewed clinical distributions) and fills categorical nulls with `'Unknown'`. Renders two visualisations: a 14-panel **histogram grid** for feature distributions and a **lower-triangle correlation heatmap** (upper triangle masked to eliminate redundancy).

### Cell 3 — Feature Engineering & Splitting
Binarises the target (`num > 0 → 1`), applies `pd.get_dummies` for one-hot encoding, and persists the resulting `feature_columns` list — critical for ensuring inference inputs align with training schema. Performs a **stratified 80/20 train-test split** (train: 736, test: 184, positive rate preserved at ~55% in both). Fits `StandardScaler` exclusively on the training set and transforms both splits to prevent data leakage. Scaler is persisted immediately after fitting.

### Cell 4 — Model Training & Evaluation
Constructs a **fully connected neural network**:

```
Input(35) → Dense(128, ReLU) → BatchNormalization
          → Dense(64,  ReLU) → BatchNormalization
          → Dense(32,  ReLU)
          → Dense(1, Sigmoid)
```

Compiled with **Adam** (`lr=1e-3`) and **binary crossentropy**. Training monitored via two callbacks: `EarlyStopping` on `val_auc` (patience=15, restores best weights) and `ReduceLROnPlateau` on `val_loss` (factor=0.5, patience=7) — allowing up to 100 epochs while preventing overfit on the small dataset. Evaluation outputs accuracy, full classification report (precision, recall, F1 per class), and a labelled confusion matrix.

### Cell 5 — Save Artifacts & Sample Prediction
Persists three inference artifacts to Google Drive: the trained model (`.keras`), the fitted scaler (`.pkl`), and the feature column schema (`.pkl`). Runs a hardcoded sample patient through the full inference pipeline and outputs both the raw probability and the binary classification result.

### Cell 6 — Interactive GUI Prediction
Replaces console `input()` with a fully styled `ipywidgets` interface injected with custom CSS (GitHub dark theme, DM Serif Display + DM Mono typography). Numeric fields use **IntSlider / FloatSlider** to constrain input ranges; categorical fields use **Dropdown** widgets matching exact training values. A two-column grid layout with a styled **Run Prediction** button triggers the full inference pipeline on click — encoding, reindexing to `feature_columns`, scaling, prediction — and renders a colour-coded result card (🟢 No Disease / 🔴 Heart Disease Detected) with raw probability and confidence score.

---

## 🗃️ Saved Artifacts

| File | Description |
|---|---|
| `heart_disease_model.keras` | Trained Keras model in native format |
| `heart_scaler.pkl` | `StandardScaler` fit on training data only |
| `heart_features.pkl` | Ordered feature list for inference schema alignment |

---

## 🚀 Run in Colab

[![Open In Colab](https://colab.research.google.com/drive/1a0p7yv38avhVELzzlh-pPjssC_KYlpAI?usp=sharing)

1. Click the badge above to open the notebook in Google Colab
2. Connect to a runtime — GPU recommended (`Runtime → Change runtime type → T4 GPU`)
3. Mount Google Drive when prompted in Cell 1
4. Run all cells in order — `Runtime → Run all` (`Ctrl+F9`)
5. Use the **interactive GUI in Cell 6** to enter patient details and receive a live prediction

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.x, Keras |
| Data & EDA | pandas, numpy, matplotlib, seaborn |
| ML Utilities | scikit-learn (StandardScaler, train_test_split, metrics) |
| Persistence | joblib |
| GUI | ipywidgets, IPython.display |
| Environment | Google Colab, Google Drive |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Ensure any new notebook cells follow the existing sequential structure and that all artifacts are re-saved after retraining.

## 📜 License

This project is licensed under the [MIT License](LICENSE).
