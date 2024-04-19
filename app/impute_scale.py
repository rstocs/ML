import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from load_data import read_data
from split_training_data import create_train_val_test_data
from clean_data import clean_data


class ImputeScale:
    def __init__(self, imputer, scaler, fitted):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = StandardScaler()
        self.fitted = False

    def impute_and_scale(self, data, non_categorical_features):
        """
        Impute and scale non-categorical features.
        Parameters:
            data (pandas df): clean encoded data to be imputed and scaled
        Returns:
            encoded_imputed_scaled_data (pandas df): encoded, imputed and scaled data
        """
        # Initialize imputed scaled data
        imputed_scaled_data = None

        # if imputer and scaler not fitted, fit and transform
        if not self.fitted:
            imputed_data = pd.DataFrame(self.imputer.fit_transform(data[non_categorical_features]),
                                        columns=data[non_categorical_features].columns)
            imputed_scaled_data = pd.DataFrame(self.scaler.fit_transform(imputed_data), columns=imputed_data.columns)
            print("imputer statistics", self.imputer.statistics_)
            print("train_imputed var", imputed_data.var())
            self.fitted = True

        # Otherwise, transform using fitted imputer and scaler
        else:
            imputed_data = pd.DataFrame(self.imputer.transform(data[non_categorical_features]),
                                        columns=data[non_categorical_features].columns)
            imputed_scaled_data = pd.DataFrame(self.scaler.transform(imputed_data), columns=imputed_data.columns)

        # Combine encoded categorical features with imputed scaled non-categorical features
        all_columns = set(data.columns)
        encoded_features = list(all_columns - set(non_categorical_features))
        encoded_imputed_scaled_data = pd.concat([imputed_scaled_data, data[encoded_features]], axis=1)
        return encoded_imputed_scaled_data


if __name__ == "__main__":
    # read_data with the correct key to retrieve the filepath
    data_train = read_data()
    print("data_train head", data_train.head())

    categorical_features = []
    non_categorical_features = []

    try:
        with open("../config/features.json") as file:
            features = json.load(file)
        categorical_features = features["CATEGORICAL_FEATURES"]
        non_categorical_features = features["NON_CATEGORICAL_FEATURES"]
        encoded_features = features["ENCODED_FEATURES"]
    except FileNotFoundError:
        print("File not found.")

    data_train = clean_data(data_train, categorical_features, save_encoded_features=True)
    print("clean train data", data_train.head())

    # split data
    train, val, test = create_train_val_test_data(data=data_train)
    print("train shape", train.shape)
    print(train.head())

    print("val shape", val.shape)
    print(val.head())

    print("test shape", test.shape)
    print(test.head())

    imputeScale = ImputeScale(imputer=None, scaler=None, fitted=False)

    # impute and encode train data
    encoded_imputed_scaled_train = imputeScale.impute_and_scale(train, non_categorical_features)
    print("encoded_imputed_scaled_train", encoded_imputed_scaled_train.head())

    # impute and encode val data
    encoded_imputed_scaled_val = imputeScale.impute_and_scale(val, non_categorical_features)
    print("encoded_imputed_scaled_val", encoded_imputed_scaled_val.head())
