import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the saved model and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('rf_model_smote.pkl', 'rb') as f:
    model = pickle.load(f)

# Label Encoding mappings (must match those used in training)
label_maps = {
    'Sex': {'male': 1, 'female': 0},
    'Housing': {'own': 2, 'free': 0, 'rent': 1},
    'Saving accounts': {'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4, 'unknown': 0},
    'Checking account': {'little': 1, 'moderate': 2, 'rich': 3, 'unknown': 0},
    'Purpose': {
        'radio/TV': 4, 'education': 0, 'furniture/equipment': 1,
        'new car': 3, 'used car': 8, 'business': 2, 'domestic appliances': 5,
        'repairs': 6, 'vacation/others': 7
    }
}

st.title("Credit Risk Cluster Prediction")
st.write("Fill the details below to predict your credit risk cluster.")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ['male', 'female'])
job = st.selectbox("Job (0=unskilled, 3=highly skilled)", [0, 1, 2, 3])
housing = st.selectbox("Housing", ['own', 'free', 'rent'])
saving = st.selectbox("Saving accounts", ['little', 'moderate', 'quite rich', 'rich', 'unknown'])
checking = st.selectbox("Checking account", ['little', 'moderate', 'rich', 'unknown'])
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=1000)
duration = st.number_input("Duration (in months)", min_value=4, max_value=72, value=12)
purpose = st.selectbox("Purpose", [
    'radio/TV', 'education', 'furniture/equipment', 'new car', 'used car',
    'business', 'domestic appliances', 'repairs', 'vacation/others']
)

if st.button("Predict Cluster"):
    # Encode categorical inputs
    sex_encoded = label_maps['Sex'][sex]
    housing_encoded = label_maps['Housing'][housing]
    saving_encoded = label_maps['Saving accounts'][saving]
    checking_encoded = label_maps['Checking account'][checking]
    purpose_encoded = label_maps['Purpose'][purpose]

    # Feature Engineering
    credit_per_month = credit_amount / duration
    is_young = 1 if age < 25 else 0

    # Prepare unscaled numeric input
    raw_input = pd.DataFrame([{
        'Age': age,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Credit_per_month': credit_per_month
    }])

    # Apply scaler to numeric columns
    scaled_values = scaler.fit_transform(raw_input)
    scaled_df = pd.DataFrame(scaled_values, columns=['Age', 'Credit amount', 'Duration', 'Credit_per_month'])

    # Create final input DataFrame
    input_df = pd.DataFrame([{
        'Age': scaled_df['Age'][0],
        'Credit amount': scaled_df['Credit amount'][0],
        'Duration': scaled_df['Duration'][0],
        'Credit_per_month': scaled_df['Credit_per_month'][0]
    }])

    # Predict
    prediction = model.predict(input_df)[0]

    # Show results
    st.success(f"Predicted Credit Risk Cluster: {prediction}")
    if prediction == 0:
        st.info("Cluster 0: Likely lower credit risk")
    else:
        st.warning("Cluster 1: Potentially higher credit risk")