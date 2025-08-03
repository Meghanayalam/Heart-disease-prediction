# 🫀 Heart Disease Prediction using Machine Learning

A comprehensive machine learning project to predict the presence of heart disease based on patient clinical features. This project demonstrates a real-world ML pipeline including data preprocessing, visualization, feature engineering, model training, evaluation, and deployment-ready model saving.



## 📌 Project Motivation

Cardiovascular diseases are among the leading causes of death worldwide. Early prediction of heart disease can significantly improve treatment outcomes. This project aims to build an interpretable and efficient classification model using the UCI Heart Disease Dataset, providing insights into the most important health indicators and their relationship with heart conditions.



## 📁 Folder Structure
heart_project/
├── heart_predict.ipynb # Google Colab notebook with complete ML pipeline
├── heart.csv # Dataset
├── heart_rf_model.pkl # Saved Random Forest model
├── scaler.pkl # Saved StandardScaler for future use
├── README.md # Project documentation
└── requirements.txt # Python dependencies




---

## 📦 Technologies Used

| Category         | Libraries/Tools                           |
|------------------|--------------------------------------------|
| Programming      | Python 3, Jupyter Notebook (Google Colab)  |
| Data Handling    | pandas, numpy                              |
| Visualization    | matplotlib, seaborn                        |
| ML Modeling      | scikit-learn (Random Forest)               |
| Model Saving     | joblib                                     |
| Deployment Prep  | Model & scaler serialized                  |

---

## 📊 Dataset Overview

- **📂 Source**: UCI Heart Disease Dataset  
- **🎯 Target Variable**: `target` (1 = Disease, 0 = No Disease)  
- **🔢 Features**:
  - `age`
  - `sex`
  - `cp` (chest pain type)
  - `trestbps` (resting blood pressure)
  - `chol` (cholesterol)
  - `fbs` (fasting blood sugar)
  - `restecg` (resting ECG)
  - `thalach` (maximum heart rate)
  - `exang` (exercise-induced angina)
  - `oldpeak` (ST depression)
  - `slope` (slope of peak exercise ST segment)
  - `ca` (number of major vessels)
  - `thal` (thalassemia)

---

## 🧪 Machine Learning Pipeline

1. **🧹 Data Cleaning & Exploration**
   - Handled categorical features
   - Addressed missing and skewed values
   - Dataset info, summary statistics, and distributions

2. **📊 Data Visualization**
   - Pie Chart: Class distribution
   - Violin Plot: Age vs heart disease
   - KDE Plots: Cholesterol levels
   - Feature Importance: Top predictors
   - Confusion Matrix: Final model evaluation

3. **🏗️ Preprocessing**
   - One-hot encoding of categorical features
   - Feature scaling using `StandardScaler`

4. **🧠 Model Training**
   - Random Forest Classifier
   - 80/20 train-test split
   - Hyperparameters: `n_estimators=100`, `random_state=42`

5. **🎯 Evaluation Metrics**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)

6. **💾 Model Deployment Ready**
   - Saved using `joblib`:
     - `heart_rf_model.pkl`
     - `scaler.pkl`


## 🔍 Results

| Metric              | Value        |
|---------------------|--------------|
| Accuracy            | ~85–90%      |
| Precision/Recall    | Balanced     |
| Interpretability    | High (top 10 features plotted) |


