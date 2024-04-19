import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from load_data import read_data


def clean_data(data, categorical_features, save_encoded_features=False):
    """
    Clean data including handling special chars and impute missing values.
    Parameters:
        data (pandas df): input data to be cleaned
        filepath (string): filepath for loading the non-categorical featuers
    Returns:
        data_cleaned (pandas df): cleaned data
    """
    # Fix money and percents and change data type to float
    fixed_data = fix_symbol_and_data_type(data)

    # impute missing values from feature mean for non-categorical data
    data_cleaned = encode_categorical_feature(fixed_data, data[categorical_features], save_encoded_features)
    return data_cleaned


def encode_categorical_feature(data, categorical_features, save_encoded_features, filepath="../config/features.json"):
    """
    Encode categorical data..
    Parameters:
        data (pandas df): input data to be encoded
        categorical_features (list): list of categorical features to be encoded
    Returns:
        data (pandas df): encoded data
    """
    encoded_features = []
    for feature in categorical_features:
        dumb = pd.get_dummies(data[feature], drop_first=True, prefix=feature, prefix_sep='_', dummy_na=True)
        data = pd.concat([data, dumb], axis=1, sort=False)
        if save_encoded_features:
            encoded_features.extend(dumb.columns)

    # Save the encoded features names
    if save_encoded_features:
        try:
            with open(filepath) as file:
                features = json.load(file)
                features.update({"ENCODED_FEATURES": encoded_features})

                # Reopen the file in write mode and write the updated data
                with open(filepath, 'w') as file:
                    json.dump(features, file, indent=4)
        except FileNotFoundError:
            print("File not found.")

    # Drop categorical features and return
    data = data.drop(columns=categorical_features)
    return data


# replace special chars and change data type
def fix_symbol_and_data_type(data):
    """
    Replace chars and change data types.
    Parameters:
        data (pandas df): input data to be processed
    Returns:
        data (pandas df): output data with specified feature changed
    """
    replace_char_from_feature(data, 'x12', '$', '')
    replace_char_from_feature(data, 'x12', ',', '')
    replace_char_from_feature(data, 'x12', ')', '')
    replace_char_from_feature(data, 'x12', '(', '-')
    replace_char_from_feature(data, 'x63', '%', '')

    # change data type to float
    change_data_type(data, 'x12', float)
    change_data_type(data, 'x63', float)

    return data


def replace_char_from_feature(data, feature, old_char, new_char):
    """
    Romove or replace characters from feature to new_char.
    Parameters:
        data (pandas df): input data to be processed
        feature (str): name of the feature being changed.
        old_char (str): old character to be replaced.
        new_char (str): new character to be changed to.
    Returns:
        data (pandas df): output data with specified feature changed
    """
    data[feature] = data[feature].str.replace(old_char, new_char)
    return data


def change_data_type(data, feature, type):
    """
    Change the data type of a feature.
    Parameters:
        data (pandas df): input data to be processed
        feature (str): name of the feature being changed.
        type (str): the type of data being changed to.
    Returns:
        data (pandas df): output data with specified feature type changed
    """
    data[feature] = data[feature].astype(type)
    return data


if __name__ == "__main__":
    # read_data with the correct key to retrieve the filepath
    data_train = read_data()
    print(data_train.head())

    categorical_features = []

    try:
        with open("../config/features.json") as file:
            features = json.load(file)
        categorical_features = features["CATEGORICAL_FEATURES"]
    except FileNotFoundError:
        print("File not found.")

    # Call
    data_train_clean = clean_data(data_train, categorical_features, save_encoded_features=True)
    print("data train cleaned", data_train_clean.shape)
    print(data_train_clean.head())  # Print the first few rows of the dataframe
    # print(test.head())



