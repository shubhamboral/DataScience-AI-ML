import streamlit as st
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load('titanic_rf_model.pkl')
le_sex = joblib.load('sex_encoder.pkl')
le_embarked = joblib.load('embarked_encoder.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict the survival probability.")

# User input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0.42, 80.0, 29.0)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.slider("Fare Paid", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Prepare input for model
if st.button("Predict Survival"):
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = le_embarked.transform([embarked])[0]

    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

    # Predict probability
    survival_proba = model.predict_proba(input_data)[0][1]  # Probability of survival
    survival_percent = round(survival_proba * 100, 2)

    if survival_percent >= 50:
        st.success(f"✅ This passenger has a **{survival_percent}% chance** of surviving.")
    else:
        st.error(f"❌ This passenger has only a **{survival_percent}% chance** of surviving.")
