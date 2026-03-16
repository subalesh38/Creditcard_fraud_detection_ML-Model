# Credit Card Fraud Detection System

## Project Description
This project implements a robust machine learning system to identify fraudulent credit card transactions. Using a Decision Tree Classifier, the system distinguishes between genuine and fraudulent activities based on anonymized transaction features.

## Key Features
*   **Data Preprocessing**: Scaling of the 'Amount' feature using StandardScaler to ensure numerical stability.
*   **Class Imbalance Handling**: Utilization of SMOTE (Synthetic Minority Over-sampling Technique) to address the severe imbalance between fraud and genuine classes.
*   **Model**: A Decision Tree Classifier trained on balanced data to provide high-accuracy predictions.

## Model Performance
Based on the project evaluation, the model achieved the following metrics:
*   **Training Score**: 1.00 (100% accuracy on resampled training data)
*   **Test Score**: 0.9976 (~99.76% accuracy on the unseen test set)

## Setup Instructions
1.  **Install Dependencies**: Run the following command to install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Model and Scaler**: Ensure decision_tree_model.pkl and scaler.pkl are present in the root directory.
3.  **Run API**: Execute python app.py to start the Flask server.

## Flask API Usage
The system provides a /predict endpoint for real-time inference.

*   **Endpoint**: POST /predict
*   **Payload**: A JSON object containing features V1 through V28 and the transaction Amount.
*   **Example Request Body**:
    ```json
    {
        "V1": -1.35, "V2": -0.07, ..., "V28": -0.02, "Amount": 149.62
    }
    ```
