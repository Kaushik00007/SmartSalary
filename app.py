import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("model.pkl") 
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.title("💼 Employee Income Prediction App")

uploaded_file = st.file_uploader("Upload employee data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df)

    # Apply preprocessor
    X_transformed = preprocessor.transform(df)

    # Predict
    predictions = model.predict(X_transformed)

    df["Prediction"] = predictions
    st.subheader("Prediction Results")
    st.dataframe(df)
