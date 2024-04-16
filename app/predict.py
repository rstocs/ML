import os
import pickle
import pandas as pd
import json
import preprocess_data as preprocess
from split_training_data import create_train_val_test_data as cd
from load_data import read_data as rd
from feature_engineer import initial_feature_selection as fs
import model

def predict_single(model, data, cutoff_prob):
    probabilites = pd.DataFrame(model.predict(data)).rename(columns={0: 'probs'})


def predict(result, data_X, data, cutoff_percent=0.75, show_statistics=False):
    """
    Make prediction from model.
    Parameters:
        result (BinaryResultsWrapper): a wrapper around the binary (logistic) regression results and provides various \
        methods and attributes for accessing and analyzing the results of the logistic regression model
        data (pandas df): imputed scaled encoded training data.
        cutoff_percent (float): cutoff percent of prob to be considered for positive outcome.
    Returns:
        outcome (pandas df): trained logistic regression model with feature selected.
        cutoff_prob (float): cutoff prob to be considered for positive outcome.
    """
    outcome = pd.DataFrame(result.predict(data_X)).rename(columns={0:'probs'})

    # print C_statistcs and prob_bin if show_statistics is set to True
    if show_statistics:
        # show C_Statistics
        c_statistics = get_c_statistics(outcome, data)
        print(c_statistics)

        # show prob_bin
        prob_bin = get_prob_bin(outcome)
        print(prob_bin)

    # Calculate the prob at the specified cutoff percentile
    cutoff_prob = outcome['probs'].quantile(cutoff_percent)

    return outcome, cutoff_prob

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

    # select features for model
    selected_features = fs(imputed_scaled_encoded_train)

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
    _, cutoff_prob = model.predict(model_all, imputed_scaled_encoded_all[selected_features], imputed_scaled_encoded_all,
                                   .75, True)
    print("cutoff_prob is: ", cutoff_prob)

    print("\nDone")





