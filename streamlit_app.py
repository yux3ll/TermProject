import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('random_forest_model.pkl')
preprocessor = joblib.load('preprocessing_pipeline.pkl')


# Define the main function
def main():
    st.title("Bank Marketing Prediction")
    st.write("Enter the customer's data to predict if they will subscribe to a term deposit.")

    # Collect user input for each feature
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
                               "retired", "self-employed", "services", "student", "technician", 
                               "unemployed", "unknown"])
    marital = st.selectbox("Marital Status", ["single", "married", "divorced", "unknown"])
    education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", 
                                           "illiterate", "professional.course", "university.degree", 
                                           "unknown"])
    default = st.selectbox("Credit in Default", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
    loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
    day_of_week = st.selectbox("Last Contact Day", ["mon", "tue", "wed", "thu", "fri"])
    poutcome = st.selectbox("Outcome of Previous Campaign", ["failure", "nonexistent", "success"])
    campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
    pdays = st.number_input("Days Since Previous Campaign Contact", min_value=-1, value=999)
    previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
    emp_var_rate = st.number_input("Employment Variation Rate", value=0.0)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.number_input("Euribor 3-Month Rate", value=1.0)
    nr_employed = st.number_input("Number of Employees", value=5000.0)

    # When the user clicks "Predict"
    if st.button("Predict"):
        # Preprocess input into the model format
        input_data = np.array([age, campaign, pdays, previous, emp_var_rate, cons_price_idx, 
                               cons_conf_idx, euribor3m, nr_employed])
        # Include categorical inputs (use the same order as in your one-hot encoding step)
        categorical_data = [job, marital, default, housing, loan, contact, day_of_week, poutcome, education]

        # Convert categorical to one-hot (using your training data mapping)
        # (Implement one-hot encoding or load a transformer for this)
        # Preprocess the input data using the loaded preprocessor
        input_df = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
            'housing': housing, 'loan': loan, 'contact': contact, 'day_of_week': day_of_week,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m,
            'nr.employed': nr_employed
        }])
        
        preprocessed_input = preprocessor.transform(input_df)
        final_input = preprocessed_input[0]

        # Combine numerical and categorical
        #final_input = np.concatenate((numerical_data, categorical_encoded), axis=0)

        # Get prediction
        prediction = model.predict([final_input])

        # Show result
        result = "Yes" if prediction == 1 else "No"
        st.success(f"The model predicts: {result}")

if __name__ == "__main__":
    main()