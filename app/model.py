import pandas as pd
import json
import preprocess_data as preprocess
import statsmodels.api as sm
import os
from load_data import read_data as rd
from split_training_data import create_train_val_test_data as cd
from sklearn.metrics import roc_auc_score
from feature_engineer import initial_feature_selection as fs

def train_model(data, features, show_model=False):
    """
    Train model.
    Parameters:
        data (pandas df): imputed scaled encoded training data.
        features (pandas df): selected features to train.
        show_model (boolean): whether to show the model or not.
    Returns:
        logistic regression model: trained logistic regression model with feature selected.
    """
    # train the logistic regression model
    logit = sm.Logit(data['y'], data[features])
    result = logit.fit()

    # print model summary if set show_model is set to True
    if show_model:
        print(result.summary())

    return result

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

def get_c_statistics(outcome, data, score_function=roc_auc_score):
    """
    Get the C-statistics.
    Parameters:
        outcome (pandas df): trained logistic regression model with feature selected.
        data (pandas df): imputed scaled encoded training data.
        score_function (function): function for calculating the roc_auc score.
    Returns:
        C-Statistics (pandas df): C-statistics.
    """
    # print("outcome[y]", outcome.shape)
    # print("data_y", data_y)
    outcome['y'] = data['y']
    c_statistics = score_function(outcome['y'], outcome['probs'])
    return c_statistics

def get_prob_bin(outcome, q=20):
    """
    Get probability bin.
    Parameters:
        outcome (pandas df): trained logistic regression model with feature selected.
        q (int): number of bins.
    Returns:
        prob_bin (pandas df): probability bin.
    """
    outcome['prob_bin'] = pd.qcut(outcome['probs'], q)
    prob_bin = outcome.groupby(['prob_bin'])['y'].sum()
    return prob_bin

if __name__ == "__main__":
    # Read training data
    data = rd("TRAIN_DATA_PATH")
    print("data is: ", data.shape)

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
    print("imputed_scaled_encoded_all is", imputed_scaled_encoded_all.head())

    # select features for model
    selected_features = fs(imputed_scaled_encoded_train)

    # train model
    model_training = train_model(imputed_scaled_encoded_train, selected_features, show_model=True)
    model_all = train_model(imputed_scaled_encoded_all, selected_features, show_model=True)
    # train_all = train_model(imputed_scaled_encoded_all, selected_features, show_model=True)
    print("\nDone")
    
    
    
