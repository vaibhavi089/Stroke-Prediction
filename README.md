# Stroke Prediction Model

A machine learning model that predicts stroke likelihood using patient health data, featuring ensemble learning and hyperparameter optimization.

## 📋 Table of Contents
- [Features](#-features)
- [Performance](#-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Future Improvements](#-future-improvements)

## ✨ Features
- Data preprocessing (missing values, feature engineering)
- SMOTEENN resampling for class imbalance
- SHAP-based feature selection
- Optuna hyperparameter tuning
- Stacking + Voting ensemble (XGBoost, LightGBM, CatBoost, SVM)

## 📊 Performance
```text
              precision    recall  f1-score   support

           0       0.99      0.98      0.98       609
           1       0.98      0.99      0.99       714

    Accuracy: 0.9856
```
## ⚙️ Installation
  ```bash
  pip install -r requirements.txt
```
First create a requirements.txt file with these contents:
```text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
imbalanced-learn>=0.9.0
optuna>=3.0.0
shap>=0.40.0
```
Then install dependencies:
```bash
 pip install -r requirements.txt
```
## 🚀 Usage
- Place your dataset as healthcare-dataset-stroke-data.csv in the project root.
- Run the Jupyter notebook:
```bash
jupyter notebook model.ipynb
```
- Key execution steps:
  1) Data loading and preprocessing
  2) Feature engineering
  3) Model training and optimization
  4) Evaluation

## 📁 Dataset
Contains 5,110 patient records with:
```text
• Demographic: age, gender
• Medical: hypertension, heart disease
• Lifestyle: smoking status, work type
• Metrics: BMI, avg_glucose_level
```
## 🧠 Methodology
1) Preprocessing:
   - Dropped 'id' column
   - Filled missing BMI values with median
   - Removed rare categories('Other' gender, 'Never_worked')
 2) Feature Engineering:
    - Created age_glucose (age × glucose level)
    - Created bmi_age_ratio (BMI / (age+1))
  3) Modelling:
     - Used stratified 80-20 train-test split
     - Applied StandardScaler
     - Selected top 15 features via SHAP
     - Optimized with 30 Optuna trials
     - Final ensemble: StackingClassifier + VotingClassifier

## 🔮 Future Improvements
- Deploy as Flask/FastAPI web service
- Add feature importance visualization
- Experiment with neural networks
- Create Docker container for deployment
  
  
  
