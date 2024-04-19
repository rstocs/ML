from flask import Flask, request, jsonify
import json
import pickle
import pandas as pd

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

def load_features(filepath, feature_key):
    # load selected features
    with open(filepath) as file:
        feature_config = json.load(file)
    selected_features = feature_config[feature_key]
    return selected_features

def load_models(filepath):
    # load the model
    with open('../models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

