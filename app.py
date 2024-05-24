import joblib
import streamlit as st
import numpy as np
import os

model_name = 'RF_Loan_model.joblib'
file_url = "https://github.com/neeraj46665/streamlit-demo/blob/main/RF_Loan_model.joblib"

# Download the model if it does not exist
if not os.path.isfile(model_name):
    import wget
    wget.download(file_url, model_name)

# Load the model with error handling
try:
    model = joblib.load(model_name)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def prediction(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0

    if Married == "Yes":
        Married = 1
    else:
        Married = 0

    if Education == "Graduate":
        Education = 0
    else:
        Education = 1

    if Self_Employed == "Yes":
        Self_Employed = 1
    else:
        Self_Employed = 0

    if Credit_History == "Outstanding Loan":
        Credit_History = 1
    else:
        Credit_History = 0

    if Property_Area == "Rural":
        Property_Area = 0
    elif Property_Area == "Semi Urban":
        Property_Area = 1
    else:
        Property_Area = 2

    Total_Income = np.log(ApplicantIncome + CoapplicantIncome)

    prediction = model.predict([[Gender, Married, Dependents, Education, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Total_Income]])
    
    if prediction == 0:
        pred = "Rejected"
    else:
        pred = "Approved"
    
    return pred

def main():
    # Front end
    st.title("Welcome to Loan Application")
    st.header("Please enter your details to proceed with your loan application")
    
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.number_input("Number of Dependents")
    Education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    ApplicantIncome = st.number_input("Applicant Income")
    CoapplicantIncome = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("Loan Amount")
    Loan_Amount_Term = st.number_input("Loan Amount Term")
    Credit_History = st.selectbox("Credit History", ("Outstanding Loan", "No Outstanding Loan"))
    Property_Area = st.selectbox("Property Area", ("Rural", "Urban", "Semi Urban"))

    if st.button("Predict"):
        result = prediction(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area)
        
        if result == "Approved":
            st.success("Your loan application is approved")
        else:
            st.error("Your loan application is rejected")

if __name__ == "__main__":
    main()
