import os
import numpy as np
import pickle
import pandas as pd
import json
import preprocess_data as preprocess
from split_training_data import create_train_val_test_data as cd
from load_data import read_data as rd
from feature_engineer import initial_feature_selection as fs
import model

def predict_single(model, data, cutoff_prob):
    probabilities = pd.DataFrame(model.predict(data)).rename(columns={0: 'probs'})
    data['probs'] = probabilities['probs']
    data['business_outcome'] = np.where(data['probs'] > cutoff_prob, 1, 0)
    return data

def prepare_data(data):
    

if __name__ == "__main__":
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

    # load the model
    with open('../models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # load selected features
    with open("../config/features.json") as file:
        feature_config = json.load(file)
    selected_features = feature_config["SELECTED_FEATURES"]

    data = predict_single(model, imputed_scaled_encoded_all[selected_features], cutoff_prob=0.5)
    print("data is; ", data)






