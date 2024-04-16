import pandas as pd
import numpy as np
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from load_data import read_data as rd
from split_training_data import create_train_val_test_data as cd

class  PreprocessData:
    def __init__(self, categorical_features, non_categorical_features, imputer, impute_training):
        self.categorical_features = []
        self.non_categorical_features = []
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.impute_training = True

## prepare data for modeling.
## Input: traing, val, or test data to be processed. 
    def prepare_data(self, data):
        """
        Prepare data for training, validation or testing.

        Parameters:
            data (pandas df): traing, validation or test data to be processed.

        Returns:
            imputed_scaled_encoded: A pandas dataframe that holds the imputed scaled encoded data.
        """
        all_columns = data.columns.tolist()

        ## Get categorical feature list from config
        with open("../config/features.json") as file:
            feature_config = json.load(file)
        self.categorical_features = feature_config["CATEGORICAL_FEATURES"]

        ## Get non categorical feature list
        self.non_categorical_features = [feature for feature in all_columns if feature != 'y' and feature not in self.categorical_features]
        imputed = self.impute_missing_values(data, self.non_categorical_features)

        imputed_scaled = self.scale_data(imputed)

        imputed_scaled_encoded = self.encode_categorical_feature(data, imputed_scaled)

        return imputed_scaled_encoded

    ## impute missing values
    def impute_missing_values(self, data, non_categorical_features):
        imputed_data = None
        if (self.impute_training):
            imputed_data = pd.DataFrame(self.imputer.fit_transform(data[non_categorical_features]),
                                        columns=data[non_categorical_features].columns)
            self.impute_training = False
        else:
            imputed_data = pd.DataFrame(self.imputer.transform(data[non_categorical_features]),
                                        columns=data[non_categorical_features].columns)
        # print("imputer statistics", imputer.statistics_)
        # print("imputed_data variance is", imputed_data.var())
        return imputed_data

    ## scale data
    def scale_data(self, data, scaler=StandardScaler()):
        train_imputed_std = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        # print("train_imputed_std", train_imputed_std)
        return train_imputed_std

    ## encode categorical data. Input: imputed_scaled_data. Output: encoded imputed scaled data
    def encode_categorical_feature(self, data, imputed_scaled_data):
        imputed_scaled_encoded = imputed_scaled_data
        for feature in self.categorical_features:
            dumb = pd.get_dummies(data[feature], drop_first=True, prefix=feature, prefix_sep='_', dummy_na=True)
            imputed_scaled_encoded = pd.concat([imputed_scaled_encoded, dumb], axis=1, sort=False)
        imputed_scaled_encoded = pd.concat([imputed_scaled_encoded, data['y']], axis=1, sort=False)
        return imputed_scaled_encoded

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

    # preprocess the data: impute missing data, standard scaling and encoding categorical features
    preprocessed_data = PreprocessData([], [], None, True)
    imputed_scaled_encoded_train = preprocessed_data.prepare_data(train)
    print(imputed_scaled_encoded_train.head())
    imputed_scaled_encoded_val = preprocessed_data.prepare_data(val)
    print(imputed_scaled_encoded_val.head())

