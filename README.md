# Stroke Prediction Model

## Overview
This project implements a machine learning model to predict the likelihood of stroke using the [Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). The model leverages advanced techniques such as feature engineering, resampling, hyperparameter optimization, and ensemble learning to achieve high accuracy. This work is part of my machine learning portfolio, showcasing skills in data preprocessing, model building, and evaluation.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Description
The goal is to build a robust classifier to predict stroke occurrences based on patient data. The notebook (`model.ipynb`) includes:
- Data preprocessing (handling missing values, encoding categorical variables)
- Feature engineering (creating interaction features like `age_glucose` and `bmi_age_ratio`)
- Handling class imbalance using SMOTEENN
- Feature selection with SHAP values
- Hyperparameter tuning with Optuna
- Ensemble modeling using stacking and voting classifiers
- Model evaluation with classification metrics

This project demonstrates proficiency in Python, machine learning libraries, and best practices for building predictive models.

## Dataset
The dataset is sourced from Kaggle and contains 5,110 records with 12 features, including:
- **Demographic**: Gender, age, marital status, residence type
- **Health**: Hypertension, heart disease, average glucose level, BMI
- **Lifestyle**: Work type, smoking status
- **Target**: Stroke (binary: 0 = No, 1 = Yes)

The dataset is imbalanced, with fewer stroke cases, addressed using SMOTEENN resampling.

## Features
- **Data Cleaning**: Removed records with `gender` as 'Other' and `work_type` as 'Never_worked'; filled missing BMI with median.
- **Feature Engineering**: Added `age_glucose` (age * glucose level) and `bmi_age_ratio` (BMI / (age + 1)).
- **Encoding**: Converted categorical variables to dummy variables.
- **Feature Selection**: Selected top 15 features using SHAP values from LightGBM.
- **Models**: Ensemble of XGBoost, LightGBM, CatBoost, and SVC, combined via stacking (Logistic Regression meta-learner) and soft voting.
- **Optimization**: Tuned XGBoost hyperparameters using Optuna over 30 trials.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stroke-prediction.git
   cd stroke-prediction
