import streamlit as st
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("churn_model.h5")

st.title("Customer Churn Prediction")

st.markdown("Enter customer details to predict if they are likely to churn.")

# Example input fields – modify based on your model's input features
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Add more fields if your model uses more features...

# Prediction button
if st.button("Predict Churn"):
    # Prepare input data as a NumPy array (shape: 1 x n_features)
    input_data = np.array([[tenure, monthly_charges, total_charges]])

    # Predict
    prediction = model.predict(input_data)[0][0]
    
    st.subheader("Result:")
    if prediction > 0.5:
        st.error(f"High chance of churn ({prediction:.2f})")
    else:
        st.success(f"Low chance of churn ({prediction:.2f})")
