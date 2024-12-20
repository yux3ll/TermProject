import streamlit as st
import pandas as pd
import joblib


def encode_month_sin_cos(X):
    month_map = {'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    months = X["month"]  # Access column by name
    months = months.map(month_map).fillna(0)  # Map and handle missing
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    return pd.DataFrame({"month_sin": month_sin, "month_cos": month_cos})

def month_feature_names_out(self, input_features): #this took way too long to figure out jesus christ
    return ["month_sin", "month_cos"]


# Load preprocessor and model
preprocessor = joblib.load('preprocessing_pipeline.pkl')
model = joblib.load('random_forest_model.pkl')  # Replace with your model file

# Define the app
st.title("Bank Term Deposit Prediction")
st.write("Enter customer details to predict if they will subscribe to a term deposit.")

# Collect user input
user_input = {
    'month': st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']),
    'job': st.selectbox('Job', ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", 
                                "self-employed", "services", "student", "technician", "unemployed", "unknown"]),
    'marital': st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown']),
    'education': st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                            'illiterate', 'professional.course', 'university.degree', 'unknown']),
    'default': st.selectbox('Credit in Default', ['no', 'yes', 'unknown']),
    'housing': st.selectbox('Housing Loan', ['no', 'yes', 'unknown']),
    'loan': st.selectbox('Personal Loan', ['no', 'yes', 'unknown']),
    'contact': st.selectbox('Contact Type', ['cellular', 'telephone']),
    'day_of_week': st.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri']),
    'poutcome': st.selectbox('Outcome of Previous Campaign', ['failure', 'nonexistent', 'success']),
    'age': st.number_input('Age', min_value=18, max_value=100, value=30),
    'campaign': st.number_input('Number of Contacts During Campaign', min_value=1, value=1),
    'pdays': st.number_input('Days Since Last Contact (-1 if not contacted)', min_value=-1, value=999),
    'previous': st.number_input('Number of Contacts Before Campaign', min_value=0, value=0),
    'emp.var.rate': st.number_input('Employment Variation Rate', value=1.1),
    'cons.price.idx': st.number_input('Consumer Price Index', value=93.5),
    'cons.conf.idx': st.number_input('Consumer Confidence Index', value=-36.4),
    'euribor3m': st.number_input('Euribor 3 Month Rate', value=4.857),
    'nr.employed': st.number_input('Number of Employees', value=5191.0)
}

# When the user clicks "Predict"
if st.button("Predict"):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Preprocess input data
    user_transformed = preprocessor.transform(user_df)
    
    # Predict
    prediction = model.predict(user_transformed)[0]
    
    # Display the prediction
    result = "Yes" if prediction == 1 else "No"
    st.subheader(f"Prediction: {result}")
