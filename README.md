🏦 Loan Default Prediction System

This project builds and deploys a Machine Learning model to predict loan default risk.
It includes data preprocessing, model training, evaluation, and a FastAPI-based REST API for real-time predictions.

📌 Project Overview

Loan default prediction is a critical task in financial risk management.
This system predicts whether a loan applicant is likely to default (1) or repay (0) based on demographic, financial, and loan-related features.

The project consists of:

A Jupyter Notebook for data analysis and model training

A trained Random Forest model

A FastAPI application to serve predictions via HTTP endpoints

📂 Repository Structure
├── ML loan Default Prediction.ipynb   # Model training & evaluation
├── loan_default_risk_data.csv         # Dataset
├── main.py                            # FastAPI application
├── rf_model.pkl                      # Trained model (generated)
├── README.md                         # Project documentation


📊 Dataset Description

The dataset contains loan application records with features such as:

age

income

loan_amount

credit_score

employment_status

marital_status

education_level

loan_term

loan_purpose

Target variable: Loan default status (0 = repay, 1 = default)


🤖 Model Training

Model development is performed in ML loan Default Prediction.ipynb and includes:

Data cleaning and preprocessing

Feature encoding for categorical variables

Train/test split

Model training using Random Forest Classifier

Model evaluation using accuracy and probability outputs

Saving the trained model using joblib


🚀 API Implementation (FastAPI)

The trained model is deployed using FastAPI (main.py) 

main
Start the API Server
uvicorn main:app --reload


Once running, the API will be available at:
http://127.0.0.1:8000


🔗 API Endpoints
🏠 Home

GET /

{
  "message": "Loan Default Prediction API is running"
}


🔮 Predict Loan Default

POST /predict

Sample Request Body
{
  "age": 56,
  "income": 22695,
  "loan_amount": 44353,
  "credit_score": 697,
  "employment_status": "Unemployed",
  "marital_status": "Divorced",
  "education_level": "Master",
  "loan_term": 12,
  "loan_purpose": "Car"
}

Sample Response
{
  "Key": "1 = likely to default, 0 = likely to repay",
  "Default_prediction": 0,
  "Default_probability": 0.243
}


🛠 Technologies Used

Python

Pandas

Scikit-learn

Joblib

FastAPI

Uvicorn

Jupyter Notebook


✅ Key Features

End-to-end ML workflow

REST API for real-time inference

Probability-based predictions

Clean and scalable architecture

Ready for deployment or extension
