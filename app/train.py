import os
import pickle

import pandas as pd
import json
import preprocess_data as preprocess
from split_training_data import create_train_val_test_data as cd
from load_data import read_data as rd
from feature_engineer import initial_feature_selection as fs
import model

def insert_features_to_json(filepath, selected_features):
    # Read the existing content of the file
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    # Update the data with the new features list
    data['SELECTED_FEATURES'] = selected_features  # This sets or updates the 'features' key

    # Write the updated data back to the JSON file
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def train_model():
    # Read training data
    data = rd("TRAIN_DATA_PATH")

    # create train, val and test data
    train, val, test = cd(data)

    # Get categorical features list
    with open("../config/features.json") as file:
        feature_config = json.load(file)
    categorical_features = feature_config["CATEGORICAL_FEATURES"]

    # preprocess the train, val and test data: impute missing data, standard scaling and encoding categorical features
    preprocessed_data = preprocess.PreprocessData([], [], None, True)
    imputed_scaled_encoded_train = preprocessed_data.prepare_data(train)
    imputed_scaled_encoded_val = preprocessed_data.prepare_data(val)
    imputed_scaled_encoded_test = preprocessed_data.prepare_data(test)

    # combine the processed train, val and test data
    imputed_scaled_encoded_all = pd.concat([imputed_scaled_encoded_train, imputed_scaled_encoded_val])
    imputed_scaled_encoded_all = pd.concat([imputed_scaled_encoded_all, imputed_scaled_encoded_test])

    # select features for model
    selected_features = fs(imputed_scaled_encoded_train)

    # save features to feature.json file
    insert_features_to_json("../config/features.json", selected_features)

    # train models with all data
    model_all = model.train_model(imputed_scaled_encoded_all, selected_features, True)

    # Save the model
    if not os.path.exists("../models"):
        # If the directory does not exist, create it
        os.makedirs("../models")
        print("Directory models was created.")

    # Saving the model to a file
    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(model_all, f)

    # Get model cutoff prob
    _, cutoff_prob = model.predict(model_all, imputed_scaled_encoded_all[selected_features], imputed_scaled_encoded_all,.75, True)
    print("cutoff_prob is: ", cutoff_prob)

    print("\nDone")


if __name__ == "__main__":
    train_model()


    
    
    
