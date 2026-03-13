import pandas as pd
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# --- File Paths ---
MODEL_PATH = 'decision_tree_model.pkl'
DATA_PATH = '/content/creditcard.csv' # Path to the original dataset for scaler fitting

# --- Load the trained model ---
decision_tree_model = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            decision_tree_model = pickle.load(file)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Please ensure the model file is in the correct directory.")
    exit() # Critical error, cannot proceed without model
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    exit()

# --- Prepare StandardScaler ---
scaler = None
model_feature_columns = None
try:
    if os.path.exists(DATA_PATH):
        # Load the original dataset to fit the scaler. This ensures consistent preprocessing.
        original_df = pd.read_csv(DATA_PATH)

        # Initialize and fit the StandardScaler on the 'Amount' column
        scaler = StandardScaler()
        scaler.fit(original_df['Amount'].values.reshape(-1, 1))
        print("StandardScaler fitted on 'Amount' column successfully.")

        # Define the exact order of feature columns expected by the model for prediction
        # Based on how 'x' was created during training: df.drop(columns=['Time', 'Amount', 'Class'])
        # This implies the model expects V1-V28 and Normalized_Amount.
        model_feature_columns = [f'V{i}' for i in range(1, 29)] + ['Normalized_Amount']
        print(f"Expected model feature columns: {model_feature_columns}")

    else:
        raise FileNotFoundError(f"Original dataset not found: {DATA_PATH}")
except FileNotFoundError as e:
    print(f"Error preparing scaler: {e}. Cannot initialize StandardScaler without the original dataset.")
    exit() # Critical error, cannot proceed without scaler
except Exception as e:
    print(f"An unexpected error occurred while preparing the scaler: {e}")
    exit()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts if a transaction is fraudulent based on input data.
    Expects a JSON payload with V1-V28 features and 'Amount'.
    Example JSON input:
    {
        "V1": 0.1, "V2": 0.2, "V3": 0.3, ..., "V28": 0.05, "Amount": 12.34
    }
    """
    if decision_tree_model is None or scaler is None or model_feature_columns is None:
        return jsonify({'error': 'API not fully initialized. Model or scaler missing.'}), 500

    try:
        data = request.get_json(force=True)

        # --- Input Validation ---
        required_v_columns = [f'V{i}' for i in range(1, 29)]
        for col in required_v_columns:
            if col not in data:
                return jsonify({'error': f'Missing required feature: {col}'}), 400
        if 'Amount' not in data:
            return jsonify({'error': 'Missing required feature: Amount'}), 400

        # --- Preprocessing ---
        # Normalize the 'Amount' using the pre-fitted scaler
        amount = data['Amount']
        normalized_amount = scaler.transform(np.array([[amount]]))[0][0] # Reshape for single sample

        # Create a list of features in the order expected by the model
        # This includes V1-V28 from the input JSON and the calculated Normalized_Amount.
        features_list = [data[f'V{i}'] for i in range(1, 29)]
        features_list.append(normalized_amount)

        # Convert the features list to a Pandas DataFrame, ensuring correct column names
        input_df_for_model = pd.DataFrame([features_list], columns=model_feature_columns)

        # --- Prediction ---
        prediction = decision_tree_model.predict(input_df_for_model)
        prediction_proba = decision_tree_model.predict_proba(input_df_for_model)

        # --- Return Results ---
        return jsonify({
            'prediction': int(prediction[0]), # 0 for genuine, 1 for fraud
            'probability_genuine': float(prediction_proba[0][0]),
            'probability_fraud': float(prediction_proba[0][1])
        })

    except KeyError as e:
        return jsonify({'error': f'Invalid input data format: Missing key {e}. Please ensure all required V-features and Amount are present.'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input data value: {e}. Please ensure numerical values are provided.'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
