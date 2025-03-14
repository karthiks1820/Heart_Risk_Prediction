import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Load background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://source.unsplash.com/1920x1080/?health,heart");
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load banner image
image = Image.open("healthcare_banner.jpg")
st.image(image, width=1920)

# Load trained model and encoders
model = pickle.load(open('xgb_best_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
had_angina_label = pickle.load(open('had_angina_label.sav', 'rb'))
had_arthritis_label = pickle.load(open('had_arthritis_label.sav', 'rb'))
sex_label = pickle.load(open('sex_label.sav', 'rb'))
age_category_ohe = pickle.load(open('age_category_ohe.sav', 'rb'))
had_diabetes_ohe = pickle.load(open('had_diabetes_ohe.sav', 'rb'))

def main():
    st.title("Heart Attack Risk Prediction")

    # User input for top 10 features
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, format="%.1f")
    had_angina = st.selectbox("Had Angina?", ["No", "Yes"])
    sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0, format="%.1f")
    physical_health_days = st.number_input("Physical Health Bad Days (past 30 days)", min_value=0, max_value=30, value=5)
    mental_health_days = st.number_input("Mental Health Bad Days (past 30 days)", min_value=0, max_value=30, value=5)
    age_category = st.selectbox("Age Category", ["Young", "Middle-Aged", "Old"])
    had_arthritis = st.selectbox("Had Arthritis?", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    had_diabetes = st.selectbox("Had Diabetes?", ["No", "Yes", "Borderline"])

    if st.button("Predict"):
        try:
            # Encode categorical inputs
            had_angina_encoded = had_angina_label.transform([had_angina])[0]
            had_arthritis_encoded = had_arthritis_label.transform([had_arthritis])[0]
            sex_encoded = sex_label.transform([sex])[0]

            # One-Hot Encoding for multi-class categorical variables
            age_encoded = age_category_ohe.transform([[age_category]])[0]
            had_diabetes_encoded = had_diabetes_ohe.transform([[had_diabetes]])[0]

            # Map encoded outputs to specific columns required by the model
            age_old = age_encoded[2] if len(age_encoded) == 3 else 0  # 'AgeCategory_Old'
            age_young = age_encoded[0] if len(age_encoded) == 3 else 0  # 'AgeCategory_Young'
            had_diabetes_no = had_diabetes_encoded[0] if len(had_diabetes_encoded) == 3 else 0  # 'HadDiabetes_No'

            # Prepare numerical input
            numerical_features = np.array([bmi, sleep_hours, physical_health_days, mental_health_days])

            # Combine all features in correct order
            final_features = np.array([
                numerical_features[0],  # BMI
                had_angina_encoded,  # HadAngina
                numerical_features[1],  # SleepHours
                numerical_features[2],  # PhysicalHealthDays
                age_old,  # AgeCategory_Old
                numerical_features[3],  # MentalHealthDays
                age_young,  # AgeCategory_Young
                had_arthritis_encoded,  # HadArthritis
                sex_encoded,  # Sex
                had_diabetes_no  # HadDiabetes_No
            ]).reshape(1, -1)

            # Scale features
            scaled_final_features = scaler.transform(final_features)

            # Make prediction
            prediction = model.predict(scaled_final_features)[0]
            risk_level = "High" if prediction == 1 else "Low"

            st.success(f"Predicted Heart Attack Risk: **{risk_level}**")

        except Exception as e:
            st.error(f"Prediction error: {e}")

main()
