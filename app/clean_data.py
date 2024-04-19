import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from load_data import read_data


def clean_data(data, filepath="../config/features.json"):
    """
    Clean data including handling special chars and impute missing values.
    Parameters:
        data (pandas df): input data to be cleaned
        filepath (string): filepath for loading the non-categorical featuers
    Returns:
        data_cleaned (pandas df): cleaned data
    """
    # Load categorical feature list and non-categorical feature list from features.json file
    try:
        with open(filepath, 'r') as file:
            features_list = json.load(file)
    except FileNotFoundError:
        print("\nFile not found")

    # Fix money and percents and change data type to float
    fixed_data = fix_symbol_and_data_type(data)

    # impute missing values from feature mean for non-categorical data
    data_cleaned = impute_missing_values(fixed_data, data[features_list["NON_CATEGORICAL_FEATURES"]])
    return data_cleaned


def encode_categorical_feature(data, categorical_features):
    """
    Encode categorical data..
    Parameters:
        data (pandas df): input data to be encoded
        categorical_features (list): list of categorical features to be encoded
    Returns:
        data (pandas df): encoded data
    """
    for feature in categorical_features:
        dumb = pd.get_dummies(data[feature], drop_first=True, prefix=feature, prefix_sep='_', dummy_na=True)
        data = pd.concat([data, dumb], axis=1, sort=False)
    data = pd.concat([data, data['y']], axis=1, sort=False)
    return data


def impute_missing_values(data, non_categorical_features,
                          imputer=SimpleImputer(missing_values=np.nan, strategy='mean')):
    """
    Impute missing data in non-categorical features from feature mean.
    Parameters:
        data (pandas df): input data to be processed
        non_categorical_features (list): list of features to be imputed
        imputer (sklearn imputer): sklearn imputer that uses mean to impute missing values
    Returns:
        imputed_data (pandas df): output data after imputing missing values
    """
    # initialize imputed_data variables for return
    imputed_data = None

    # If imputer is not fitted, fit and transform
    if not hasattr(imputer, 'statistics_'):
        # Fit and impute missing data from mean
        imputed_data = pd.DataFrame(imputer.fit_transform(data[non_categorical_features]),
                                    columns=data[non_categorical_features].columns)
    # If imputer is fitted, impute missing data from mean without fitting
    else:
        imputed_data = pd.DataFrame(imputer.transform(data[non_categorical_features]),
                                    columns=data[non_categorical_features].columns)

    # Return imputed data
    return imputed_data


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

    # Call
    data_train_clean = fix_symbol_and_data_type(data_train)
    print(data_train_clean.head())  # Print the first few rows of the dataframe
    # print(test.head())
