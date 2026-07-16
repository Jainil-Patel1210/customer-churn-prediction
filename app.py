import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter the customer's information below to predict "
    "whether they are likely to churn."
)

st.divider()

# -----------------------------
# Customer Information
# -----------------------------
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col3:
    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

# -----------------------------
# Internet Services
# -----------------------------
st.divider()
st.subheader("🌐 Internet Services")

col1, col2, col3 = st.columns(3)

with col1:
    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

# -----------------------------
# Contract and Billing
# -----------------------------
st.divider()
st.subheader("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:
    Contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col3:
    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True
):

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    prediction = model.predict(customer_data)[0]

    churn_probability = model.predict_proba(
        customer_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "⚠️ This customer is likely to churn."
        )
    else:
        st.success(
            "✅ This customer is likely to stay."
        )

    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    st.progress(float(churn_probability))