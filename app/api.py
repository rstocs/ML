# api.py
import json
import pickle

import pandas as pd
from flask import Flask, request, jsonify
from split_training_data import replace_chars

app = Flask(__name__)

train_model()

# load model
def load_model(filepath):
    # load the model
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
model = load_model("../models/model.pickle")

# load feature
def load_features(filepath, feature_key):
    # load selected features
    with open(filepath) as file:
        feature_config = json.load(file)
    return feature_config[feature_key]
selected_features = load_features("../config/features.json", "SELECTED_FEATURES")

# Establish the percentile cutoff for determining business outcomes
PERCENTILE_CUTOFF = 75

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        json_data = request.get_json()
        # Handle both single and batch JSON inputs
        if isinstance(json_data, dict):  # Single data point
            json_data = [json_data]  # Make it a list for consistent processing

        data_df = pd.DataFrame(json_data)
        # Ensure the DataFrame only contains the features used by the model

        if not all(col in data_df.columns for col in selected_features):
            return jsonify({'error': 'Missing features'}), 400

        # Preprocess and predict
        data_preprocessed = preprocess_data(data_df[features])
        probabilities = predict(model, data_preprocessed)

        # Determine the cutoff probability for the top 25%
        cutoff_probability = get_percentile_cutoff(probabilities, PERCENTILE_CUTOFF)

        # Prepare response
        responses = []
        for idx, prob in enumerate(probabilities):
            business_outcome = 'event' if prob >= cutoff_probability else 'no event'
            response = {
                'phat': prob,
                'business_outcome': business_outcome
            }
            # Include all model inputs and sort keys alphabetically
            model_inputs = {feature: json_data[idx][feature] for feature in sorted(features)}
            response.update(model_inputs)

            responses.append(response)

        return jsonify(responses)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



def prepare_data(json_data):
    # Convert single instance data point
    if isinstance(json_data, dict):  # Single data point
        json_data = [json_data]  # Make it a list for consistent processing

    # convert json data to a dataframe
    data_df = pd.DataFrame(json_data)

    # load selected feature names for training
    selected_features = load_features("../config/features.json", "SELECTED_FEATURES")

    # Ensure the DataFrame only contains the features used by the model
    if not all(col in data_df.columns for col in selected_features):
        return jsonify({'error': 'Missing features'}), 400





if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1313)
