from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Define the FastAPI app
app = FastAPI()

# Load the trained model and any preprocessing objects (e.g., scalers, encoders)
model = joblib.load('Data/best_model.pkl')
scaler = joblib.load('Data/scaler.pkl')
label_encoders = joblib.load('Data/label_encoders.pkl')

# Define the data model for input validation
class CustomerData(BaseModel):
    Age: int
    Gender: str
    ContractType: str
    MonthlyCharges: float
    TotalCharges: float
    TechSupport: str
    InternetService: str
    Tenure: int
    PaperlessBilling: str
    PaymentMethod: str
    AverageMonthlyCharges: float
    CustomerLifetimeValue: float

@app.post("/predict/")
async def predict(data: CustomerData):
    # Encode categorical features using the label encoders
    gender_encoded = label_encoders['Gender'].transform([data.Gender])[0]
    contract_type_encoded = label_encoders['ContractType'].transform([data.ContractType])[0]
    tech_support_encoded = label_encoders['TechSupport'].transform([data.TechSupport])[0]
    internet_service_encoded = label_encoders['InternetService'].transform([data.InternetService])[0]
    paperless_billing_encoded = label_encoders['PaperlessBilling'].transform([data.PaperlessBilling])[0]
    payment_method_encoded = label_encoders['PaymentMethod'].transform([data.PaymentMethod])[0]

    # Convert the input data to a NumPy array
    data_array = np.array([
        data.Age,
        gender_encoded,
        contract_type_encoded,
        data.MonthlyCharges,
        data.TotalCharges,
        tech_support_encoded,
        internet_service_encoded,
        data.Tenure,
        paperless_billing_encoded,
        payment_method_encoded,
        data.AverageMonthlyCharges,
        data.CustomerLifetimeValue
    ])

    # Preprocess the input data (e.g., scaling)
    data_scaled = scaler.transform([data_array])

    # Make predictions using the trained model
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[:, 1]

    # Return the prediction result
    return {
        "Churn": bool(prediction[0]),
        "Probability": float(probability[0])
    }

# To run the FastAPI app, use the following command:
# uvicorn app:app --reload