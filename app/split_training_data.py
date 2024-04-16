import pandas as pd
from sklearn.model_selection import train_test_split

from load_data import read_data as rd

## Creating the train/val/test set
def create_train_val_test_data(data):
    # Fixing the money and percents 
    replace_chars(data)
    # Creating the train/val/test set
    x_train, x_val, y_train, y_val = split_data(data.drop(columns=['y']), data['y'])
    x_train, x_test, y_train, y_test = split_data(x_train, y_train, test_size=4000)
    # Smashing sets back together
    train = combine_x_and_y(x_train, y_train)
    # print("train is: ", train.head())
    val = combine_x_and_y(x_val, y_val)
    test = combine_x_and_y(x_test, y_test)
    return train, val, test

## Fixing the money and percents
def replace_chars(data):
    replace_char_from_feature(data, 'x12', '$', '')
    replace_char_from_feature(data, 'x12', ',', '')
    replace_char_from_feature(data, 'x12', ')', '')
    replace_char_from_feature(data, 'x12', '(', '-')
    change_data_type(data, 'x12', float)
    replace_char_from_feature(data, 'x63', '%','')
    change_data_type(data, 'x63', float)
    return data

## split data into two subsets
def split_data(X, y, test_size=0.1, random_state=13):
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_val, y_train, y_val

## replace char in feature with new char
def replace_char_from_feature(data, feature, old_char, new_char):
    data[feature] = data[feature].str.replace(old_char, new_char)
    return data

## change the data type of a feature
def change_data_type(data, feature, type):
    data[feature] = data[feature].astype(type)
    return data

## combine x and y
def combine_x_and_y(X, y, axis=1, sort=False, drop=True):
    return pd.concat([X, y], axis=1, sort=False).reset_index(drop=True)

if __name__ == "__main__":
    # filepath key that exist in SETTINGS.json
    data = rd("TRAIN_DATA_PATH")
    # Call read_data with the correct key to retrieve the filepath
    train, val, test = create_train_val_test_data(data)
    print(train.head())  # Print the first few rows of the dataframe
    # print(val.head())
    # print(test.head())
