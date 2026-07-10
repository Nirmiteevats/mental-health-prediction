"""
Mental Health Condition Predictor App.

This module runs a Streamlit web application that allows users to input
demographic, lifestyle, and psychological factors to predict their
mental health condition using a pre-trained Random Forest model.
"""

import pickle
import numpy as np
import streamlit as st


def load_artifacts():
    """Load the trained model and label encoders from disk."""
    with open("models/mental_health_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/le_gender.pkl", "rb") as f:
        le_gender = pickle.load(f)

    with open("models/le_occupation.pkl", "rb") as f:
        le_occupation = pickle.load(f)

    with open("models/le_target.pkl", "rb") as f:
        le_target = pickle.load(f)

    return model, le_gender, le_occupation, le_target


def main():
    """Main function to run the Streamlit app."""
    # Load model and encoders
    try:
        model, le_gender, le_occupation, le_target = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure the models/ directory exists.")
        return

    st.title("Mental Health Condition Predictor")
    st.write("Enter the details below to predict the mental health condition:")

    # Feature input sliders and selectboxes
    age = st.slider("Age", 15, 60, 22)
    gender = st.selectbox("Gender", le_gender.classes_)
    occupation = st.selectbox("Occupation", le_occupation.classes_)
    sleep_hours = st.slider("Sleep Hours (per night)", 3.0, 12.0, 7.0)
    sleep_quality = st.slider("Sleep Quality (1-10)", 1.0, 10.0, 6.0)
    social_media_hours = st.slider("Social Media Hours (per day)", 0.0, 10.0, 3.0)
    academic_work_pressure = st.slider("Academic/Work Pressure (1-10)", 1.0, 10.0, 5.0)
    physical_activity_days = st.slider("Physical Activity (days per week)", 0.0, 7.0, 3.0)
    stress_level = st.slider("Stress Level (1-10)", 1.0, 10.0, 5.0)
    anxiety_score = st.slider("Anxiety Score (1-10)", 1.0, 10.0, 5.0)
    depression_score = st.slider("Depression Score (1-10)", 1.0, 10.0, 5.0)
    work_life_balance = st.slider("Work-Life Balance (1-10)", 1.0, 10.0, 5.0)
    mood_score = st.slider("Mood Score (1-10)", 1.0, 10.0, 5.0)
    concentration_level = st.slider("Concentration Level (1-10)", 1.0, 10.0, 5.0)
    social_support = st.slider("Social Support (1-10)", 1.0, 10.0, 5.0)

    # Encode categorical variables
    gender_encoded = le_gender.transform([gender])[0]
    occupation_encoded = le_occupation.transform([occupation])[0]

    # Combine all features into an array for prediction
    features = np.array([[
        age, gender_encoded, occupation_encoded, sleep_hours,
        sleep_quality, social_media_hours, academic_work_pressure,
        physical_activity_days, stress_level, anxiety_score,
        depression_score, work_life_balance, mood_score,
        concentration_level, social_support
    ]])

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(features)
        condition = le_target.inverse_transform(prediction)[0]
        st.success(f"The predicted mental health condition is: **{condition}**")


if __name__ == "__main__":
    main()
