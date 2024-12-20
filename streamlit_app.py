import streamlit as st
import pandas as pd
import joblib
import numpy as np

def encode_month_sin_cos(X):
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    months = X["month"]
    months = months.map(month_map).fillna(0)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    return pd.DataFrame({"month_sin": month_sin, "month_cos": month_cos})

def month_feature_names_out(self, input_features):
    return ["month_sin", "month_cos"]

def load_models():
    preprocessor = joblib.load('preprocessing_pipeline.pkl')
    model = joblib.load('model.pkl')
    return preprocessor, model

def get_user_input():
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
    return user_input

def predict(preprocessor, model, user_input):
    user_df = pd.DataFrame([user_input])
    user_df['y'] = 0  # Dummy target variable for preprocessing
    user_transformed = preprocessor.transform(user_df)
    user_transformed = pd.DataFrame(user_transformed, columns=preprocessor.get_feature_names_out()).iloc[:, :-1]
    prediction = model.predict(user_transformed)[0]
    prediction_prob = model.predict_proba(user_transformed)[0][1] if hasattr(model, "predict_proba") else None
    return prediction, prediction_prob

def main():
    st.title("Bank Term Deposit Prediction")
    st.write("Enter customer details to predict if they will subscribe to a term deposit.")

    preprocessor, model = load_models()
    user_input = get_user_input()

    if st.button("Predict"):
        try:
            prediction, prediction_prob = predict(preprocessor, model, user_input)
            result = "Yes" if prediction == 1 else "No"
            st.subheader(f"Prediction: {result}")
            if prediction_prob is not None:
                st.write(f"Probability of subscribing: {prediction_prob:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()