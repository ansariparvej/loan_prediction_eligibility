import streamlit as st
import pandas as pd
import pickle

file_path = '/app/loan_eligibility_model.pkl'
# Load the pickled model
model = pickle.load(open(file_path, 'rb'))

# Define the loan eligibility prediction function


def predict_loan_eligibility(data):
    # Perform the loan eligibility prediction using the loaded model
    prediction = model.predict(data)
    return prediction


def main():
    # Set the page title
    st.title('Loan Eligibility Prediction')

    # Create input fields for loan details
    gender = st.selectbox('Gender', ['Male', 'Female'])
    if gender == 'Male':
        gender = 1
    else:
        gender = 0

    married = st.selectbox('Marital Status', ['No', 'Yes'])
    if married == 'No':
        married = 0
    else:
        married = 1

    dependents = st.selectbox('Dependents', ['0', '1', '2', '3'])
    if dependents == '0':
        dependents = 0
    elif dependents == '1':
        dependents = 1
    elif dependents == '2':
        dependents = 2
    else:
        dependents = 3

    education = st.selectbox('Education', ['Not Graduate', 'Graduate'])
    if education == 'Not Graduate':
        education = 0
    else:
        education = 1

    self_employed = st.selectbox('Self Employed', ['No', 'Yes'])
    if self_employed == 'No':
        self_employed = 0
    else:
        self_employed = 1

    applicant_income = st.slider('Applicant Income', 1000, 150000)
    coapplicant_income = st.slider('Coapplicant Income', 1000, 100000)
    loan_amount = st.slider('Loan Amount', 500, 1000)
    loan_amount_term = st.slider('Loan Amount Term', 6, 480)
    loan_amount_log = st.slider('Loan Amount Log', 1, 5)
    credit_history = st.selectbox('Credit History', [0, 1])
    property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
    if property_area == 'Rural':
        property_area = 0
    elif property_area == '2':
        property_area = 1
    else:
        property_area = 2

    # Prepare the loan data as input for prediction
    data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],
        'LoanAmount_log': [loan_amount_log],

    })

    # Handle form submission
    if st.button('Predict'):
        # Perform loan eligibility prediction
        prediction = predict_loan_eligibility(data)

        # Display the prediction result
        if prediction[0] == 1:
            st.success('Congratulations! The loan is likely to be approved.')
        else:
            st.warning('Sorry, the loan is unlikely to be approved.')


if __name__ == '__main__':
    main()
