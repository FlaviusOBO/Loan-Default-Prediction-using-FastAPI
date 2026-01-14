# main.py
# FastAPI app for loan default prediction
# run with: uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load("rf_model.pkl")

# Create app
app = FastAPI(title="Loan Default Prediction API")

# Input schema
class LoanApplication(BaseModel):
    age: int
    income: int
    loan_amount: int
    credit_score: int
    employment_status: str
    marital_status: str
    education_level: str
    loan_term: int
    loan_purpose: str

    class Config:
        json_schema_extra = {
            "example": {
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
        }


@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}

@app.post("/predict")
def predict_loan_default(data: LoanApplication):
    try:
        loan_pred_df = pd.DataFrame([data.dict()])
        prediction = model.predict(loan_pred_df)[0]
        probability = model.predict_proba(loan_pred_df)[0][1]

        return {
            "Key": "1 = likely to default, 0 = likely to repay",
            "Default_prediction": int(prediction),
            "Default_probability": round(float(probability), 3)
        }
    except Exception as e:
        return {"error": str(e)}
