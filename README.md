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
requirements.txt
```
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
##Usage
  
