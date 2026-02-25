import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load Model, Scaler, and Encoder
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Liver Disease Multi-Class Predictor",
    layout="centered"
)

st.title("🩺 Liver Disease Multi-Class Predictor")
st.success("Model and features loaded successfully!")

st.header("Enter Patient Data")

# -------------------------------
# Inputs (MUST MATCH TRAINING ORDER)
# -------------------------------

age = st.number_input("Age", min_value=0.0)
sex = st.selectbox("Sex", ["Male", "Female"])
albumin = st.number_input("Albumin")
alkaline_phosphatase = st.number_input("Alkaline Phosphatase")
alanine_aminotransferase = st.number_input("Alanine Aminotransferase")
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
bilirubin = st.number_input("Bilirubin")
cholinesterase = st.number_input("Cholinesterase")
cholesterol = st.number_input("Cholesterol")
creatinina = st.number_input("Creatinina")
gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase")
protein = st.number_input("Protein")

# Encode categorical variable
sex_encoded = 1 if sex == "Male" else 0

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    input_data = np.array([[ 
        age,
        sex_encoded,
        albumin,
        alkaline_phosphatase,
        alanine_aminotransferase,
        aspartate_aminotransferase,
        bilirubin,
        cholinesterase,
        cholesterol,
        creatinina,
        gamma_glutamyl_transferase,
        protein
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict encoded value
    prediction_encoded = model.predict(input_scaled)

    # Decode prediction
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    # Get probabilities
    probabilities = model.predict_proba(input_scaled)

    # Decode class labels for probability table
    decoded_classes = label_encoder.inverse_transform(model.classes_)

    st.subheader("Prediction Result")
    st.success(f"Predicted diagnosis: {prediction_label[0]}")

    st.subheader("Class Probabilities")

    prob_df = pd.DataFrame({
        "Class": decoded_classes,
        "Probability": probabilities[0]
    })

    st.dataframe(prob_df, use_container_width=True)