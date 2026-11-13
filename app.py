
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load saved files
model = joblib.load("model/decision_tree_model.pkl")
encoder = joblib.load("model/target_encoder.pkl")
features = joblib.load("model/feature_columns.pkl")

# Page settings
st.set_page_config(
    page_title="Genetic Disorder Risk Assessment",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Explainable Genetic Disorder Risk Assessment System")

st.write(
    "Predict genetic disorder categories using a Decision Tree model."
)

st.info(
    "Educational project only. This prediction is not a medical diagnosis."
)

# Patient input
st.header("Patient Information")

st.write("Enter the encoded feature values for prediction.")

input_values = {}

for feature in features:
    input_values[feature] = st.number_input(
        feature,
        value=0.0,
        key=feature
    )

# Prediction
if st.button("Predict Genetic Disorder"):

    input_data = pd.DataFrame(
        [input_values],
        columns=features
    )

    prediction = model.predict(input_data)

    disorder = encoder.inverse_transform(prediction)

    st.success("Prediction Completed!")

    st.subheader("Predicted Genetic Disorder")

    st.write("###", disorder[0])

# Feature importance
st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

st.bar_chart(
    importance.set_index("Feature")
)

st.header("Model Performance")

st.write("Training Accuracy: 61.57%")
st.write("Testing Accuracy: 59.64%")
st.write("Cross-validation Macro F1 Score: 51.00%")
