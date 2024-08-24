import requests

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predict/"

# Sample customer data to send in the POST request
customer_data = {
    "Age": 35,                    
    "Gender": "Male",             # Gender: 'Male' or 'Female'
    "ContractType": "Month-to-month",  # Contract type: 'Month-to-month', 'One Year', 'Two Year'
    "MonthlyCharges": 70.5,       
    "TotalCharges": 1230.50,      
    "TechSupport": "Yes",         # Tech support: 'Yes' or 'No'
    "InternetService": "Fiber optic",  # Internet service: 'DSL', 'Fiber optic', 'No'
    "Tenure": 24,                 # In Months
    "PaperlessBilling": "Yes",    # Paperless billing: 'Yes' or 'No'
    "PaymentMethod": "Credit card (automatic)",  # Payment method: 'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
    "AverageMonthlyCharges": 72.5, 
    "CustomerLifetimeValue": 5000  
}


# Send the POST request
response = requests.post(url, json=customer_data)

# Print the response from the server
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print(f"Failed to get a prediction. Status code: {response.status_code}")
    print("Response:", response.text)