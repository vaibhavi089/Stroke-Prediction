# Stroke Prediction Model

This project focuses on predicting the likelihood of a stroke using a healthcare dataset. The model employs advanced machine learning techniques, including ensemble methods and hyperparameter optimization, to achieve high accuracy.

## Features

- **Data Preprocessing**: Handles missing values, removes irrelevant columns, and filters out rare categories.
- **Feature Engineering**: Creates new features like `age_glucose` and `bmi_age_ratio` to enhance predictive power.
- **Resampling**: Uses SMOTEENN to address class imbalance in the dataset.
- **Feature Selection**: Utilizes SHAP values to identify the most important features.
- **Hyperparameter Optimization**: Employs Optuna to fine-tune model parameters for optimal performance.
- **Ensemble Learning**: Combines multiple models (XGBoost, LightGBM, CatBoost, SVM) using stacking and voting classifiers for robust predictions.

## Model Performance

The final model achieves the following performance metrics:
          precision    recall  f1-score   support

       0       0.99      0.98      0.98       609
       1       0.98      0.99      0.99       714

accuracy                           0.99      1323
macro avg 0.99 0.99 0.99 1323
weighted avg 0.99 0.99 0.99 1323

Final Optimized Accuracy: 0.9856


## Requirements

To run this project, you'll need the following Python libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- imbalanced-learn
- optuna
- shap

You can install all required packages using:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn optuna shap



