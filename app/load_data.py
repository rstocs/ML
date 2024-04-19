import json
import pandas as pd
import numpy as np
from collections import Counter


def convert_json_data_to_df(json_data):
    # Handle both single and batch JSON inputs
    if isinstance(json_data, dict):  # Single data point
        json_data = [json_data]  # Make it a list for consistent processing

    data_df = pd.DataFrame(json_data)
    return data_df


def read_data(filepath="../config/SETTINGS.json", key="TRAIN_DATA_PATH"):
    """
    read data from filepath specified in a SETTING.json file.
    Parameters:
        filepath (str): filepath of the json file that contains the path of the data.
        key (str): the key in the json file to the path of the data. default: 'TRAIN_DATA_PATH'
    Returns:
        data_frame (pandas df): data frame read from filepath.
    """
    with open(filepath) as file:
        settings = json.load(file)
    return pd.read_csv(settings[key])


def save_features(data, categorical_features):
    # Convert columns to a set and remove 'y' and categorical features to get non-categorical features
    all_features = data.drop(columns=['y'])
    print(type(all_features))
    non_categorical_features = list(all_features.drop(columns=categorical_features).columns)
    print("non_categorical_features", type(non_categorical_features))

    # insert categorical features to features.json file
    insert_features_to_json(
        {"CATEGORICAL_FEATURES": categorical_features, "NON_CATEGORICAL_FEATURES": non_categorical_features})


def insert_features_to_json(feature_dict, filepath="../config/features.json"):
    """
    Insert feature_key and feature_value to SETTING.json file.
    Parameters:
        feature_dict (dict): the dictionary that contains the key/value pair to be inserted. feature_key (str): the
        feature key specified in the SETTING.json file. Example of feature key includes
        "CATEGORICAL_FEATURES", "NON_CATEGORICAL_FEATURES", "SELECTED_FEATURES", "NUM_OF_FEATURES_SELECTED",
        etc. feature_value (list[str]): list of feature names to be inserted to SETTING.json file.
        filepath (str): filepath of the SETTING.json file.
    Returns:
        NONE.
    """
    # Read the existing content of the file
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            data.update(feature_dict)

            # Reopen the file in write mode and write the updated data
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)

    except FileNotFoundError:
        print("File not found.")


def describe_target(data):
    Counter(data.y)
    print("\ncount y is: ", Counter(data.y))


def get_non_float_features(data):
    """
    Get list of features that contain non-float values.
    Parameters:
        data (pandas df): data to be examined.
    Returns:
        object_features: list of features that contain object values.
        int_features: list of features that contain integer values.
    """
    object_features = data.columns[data.dtypes == 'object'].tolist()
    int_features = data.drop('y', axis=1).columns[data.drop('y', axis=1).dtypes == 'int'].tolist()
    return object_features, int_features


def investigate_object(data):
    """
    This function prints the unique categories of all the object dtype columns.
    It prints '...' if there are more than 13 unique categories.
    """
    col_obj = data.columns[data.dtypes == 'object']

    for i in range(len(col_obj)):
        if len(data[col_obj[i]].unique()) > 13:
            print(col_obj[i] + ":", "Unique Values:", np.append(data[col_obj[i]].unique()[:13], "..."))
        else:
            print(col_obj[i] + ":", "Unique Values:", data[col_obj[i]].unique())

    del col_obj


if __name__ == "__main__":
    # read training data with the default filepath and key
    data_train = read_data()
    print(data_train.head())  # Print the first few rows of the dataframe

    # Describe target variable
    describe_target(data_train)

    # get and print object_features and int_features
    object_features, int_features = get_non_float_features(data_train)
    print("\nobject dtype: ", object_features)
    print("int_64 dtype: ", int_features)
    print("The rest of the columns have float64 dtypes.\n")

    # prints the unique categories of all the object dtype columns
    investigate_object(data_train)

    # Insert features to json
    categorical_features = ['x5', 'x31', 'x81', 'x82']
    save_features(data_train, categorical_features)
