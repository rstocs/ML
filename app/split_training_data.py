import pandas as pd
from sklearn.model_selection import train_test_split
from load_data import read_data

## Creating the train/val/test set
def create_train_val_test_data(data):
    # Creating the train/val/test set
    x_train, x_val, y_train, y_val = split_data(data.drop(columns=['y']), data['y'])
    x_train, x_test, y_train, y_test = split_data(x_train, y_train, test_size=4000)
    # Smashing sets back together
    train = combine_x_and_y(x_train, y_train)
    # print("train is: ", train.head())
    val = combine_x_and_y(x_val, y_val)
    test = combine_x_and_y(x_test, y_test)
    return train, val, test

## split data into two subsets
def split_data(X, y, test_size=0.1, random_state=13):
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_val, y_train, y_val

## combine x and y
def combine_x_and_y(X, y, axis=1, sort=False, drop=True):
    return pd.concat([X, y], axis=1, sort=False).reset_index(drop=True)

if __name__ == "__main__":
    # filepath key that exist in SETTINGS.json
    data = read_data()
    print(data.shape)
    # Call read_data with the correct key to retrieve the filepath
    train, val, test = create_train_val_test_data(data)
    print(train.shape)  # Print the first few rows of the dataframe
    print(val.shape)
    print(test.shape)
