# Heart-Disease-Predictor


This project demonstrates how to use **Machine Learning algorithms** to predict the likelihood of presence of **heart disease** based on patient medical records, assisting in the early diagnosis.  
The dataset used is the **UCI Heart Disease dataset**, and models like **Logistic Regression** and **Random Forest** are trained, evaluated, and deployed.


---
## 📊 Dataset
- **Source:** UCI Machine Learning Repository – Heart Disease dataset  
- **Features include:**
  - Age, Sex, Chest Pain Type (cp)  
  - Resting Blood Pressure (trestbps)  
  - Cholesterol (chol)  
  - Fasting Blood Sugar (fbs)  
  - Resting ECG (restecg)  
  - Maximum Heart Rate (thalch)  
  - Exercise-induced Angina (exang)  
  - Oldpeak, Slope, ca, thal  
- **Target:** `num` → Converted into binary classification:  
  - `0` → No Heart Disease  
  - `1` → Heart Disease Present  

---

## ⚙️ Project Workflow
1. **Exploratory Data Analysis (EDA)**  
   - Checked for missing values  
   - Plotted feature distributions  
   - Generated correlation heatmap  

2. **Data Preprocessing**  
   - Filled missing numeric values with mean  
   - One-Hot Encoding for categorical features  
   - Standardization with `StandardScaler`  
   - Train-Test Split (80:20)  

3. **Model Training**  
   - Logistic Regression → Accuracy ~84%  
   - Random Forest → Accuracy ~88% (best model)  

4. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix visualization  
   - Feature Importance (Random Forest)  

5. **Deployment Preparation**  
   - Saved trained model: `heart_rf_model.pkl`  
   - Saved scaler: `heart_scaler.pkl`  
   - Saved feature columns: `heart_feature_columns.pkl`  
   - Created sample user template CSV: `Heart_user_template.csv`  

6. **Prediction**  
   - Loads new user input CSV  
   - Preprocesses data using same pipeline  
   - Predicts `Heart_Disease_Prediction` (0 or 1)  

---

## 🚀 Installation & Setup

Clone the repo:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

```




## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.  

## 📜 License
This project is licensed under the MIT License.

