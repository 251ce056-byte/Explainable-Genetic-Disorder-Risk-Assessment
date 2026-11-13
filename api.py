
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/decision_tree_model.pkl")
encoder = joblib.load("model/target_encoder.pkl")
features = joblib.load("model/feature_columns.pkl")

@app.route("/")
def home():
    return "Genetic Disorder Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_data)
    disorder = encoder.inverse_transform(prediction)

    return jsonify({
        "Predicted Genetic Disorder": disorder[0]
    })

if __name__ == "__main__":
    app.run()
