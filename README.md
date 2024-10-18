![Diabetes Prediction](https://github.com/kmoreno013/ML-Model-for-Diabetes-Prediction/blob/main/diabetes.png)

# ML Model for Diabetes Prediction

## Overview
This project focuses on detecting individuals at risk of diabetes, a chronic condition characterized by high blood sugar levels that can lead to serious health complications like heart disease, kidney failure, and nerve damage. The model uses patient data to predict the likelihood of developing diabetes, which can be crucial for early intervention and lifestyle changes. Given the increasing prevalence of diabetes worldwide, such models offer a valuable tool for healthcare providers in managing and preventing the disease effectively.

## About the Dataset
The dataset from Mendeley titled "Early-stage diabetes risk prediction dataset" contains medical records aimed at predicting diabetes risk. It includes attributes like glucose levels, BMI, blood pressure, and other health indicators from a sample population. This dataset is valuable for building machine learning models to predict the onset of diabetes by analyzing patterns in these attributes. It helps researchers and healthcare providers identify key risk factors for diabetes and develop early intervention strategies.

For more details, you can access the dataset here:  [Early-stage diabetes risk prediction dataset](https://data.mendeley.com/datasets/wj9rwkp9c2/1)

## Data Cleaning
In this phase, missing values, duplicates, and inconsistencies in the dataset were addressed. Data types were verified and converted as needed to ensure the integrity of the dataset.

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution of features and their relationships with the target variable (diabetes risk). Key visualizations included histograms, scatter plots, and correlation matrices to identify patterns and potential predictors.

## Data Preprocessing
Data preprocessing involved:
* Handling missing values through imputation.
* Normalizing or standardizing numerical features.
* Encoding categorical variables using one-hot encoding.
* Splitting the dataset into training and testing sets.

## Machine Learning Model Development
Various machine learning models were developed and tuned, including:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Support Vector Classifier (SVC)
* Random Forest Classifier
* Decision Tree Classifier
* XGBoost Classifier
Hyperparameter tuning was performed using GridSearchCV to optimize model performance.

## Evaluation and Selection
Models were evaluated based on accuracy, F1-score, and Jaccard score. The results were compared to identify the best-performing model.

## Best Model
The Random Forest Classifier achieved the best performance, with:
* Accuracy: 0.990
* F1-Score: 0.990
* Jaccard Score: 0.981
