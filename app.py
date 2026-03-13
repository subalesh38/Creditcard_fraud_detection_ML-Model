import pandas as pd
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# File paths
MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"

# Load model
try:
    with open(MODEL_PATH, "rb") as file:
        decision_tree_model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    decision_tree_model = None

# Load scaler
try:
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Expected feature order
model_feature_columns = [f"V{i}" for i in range(1, 29)] + ["Normalized_Amount"]


@app.route("/")
def home():
    return jsonify({"message": "Credit Card Fraud Detection API is running"})


@app.route("/predict", methods=["POST"])
def predict():

    if decision_tree_model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    try:
        data = request.get_json()

        # Validate input
        required_v_columns = [f"V{i}" for i in range(1, 29)]
        for col in required_v_columns:
            if col not in data:
                return jsonify({"error": f"Missing feature {col}"}), 400

        if "Amount" not in data:
            return jsonify({"error": "Missing feature Amount"}), 400

        # Normalize amount
        amount = data["Amount"]
        normalized_amount = scaler.transform(np.array([[amount]]))[0][0]

        # Prepare features
        features = [data[f"V{i}"] for i in range(1, 29)]
        features.append(normalized_amount)

        input_df = pd.DataFrame([features], columns=model_feature_columns)

        # Prediction
        prediction = decision_tree_model.predict(input_df)
        prediction_proba = decision_tree_model.predict_proba(input_df)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_genuine": float(prediction_proba[0][0]),
            "probability_fraud": float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


