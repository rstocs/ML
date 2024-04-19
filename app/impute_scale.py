import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from load_data import read_data


class ImputeScale:
    def __init__(self, non_categorical_features, imputer, scaler, fitted):
        self.non_categorical_features = non_categorical_features
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.scaler = StandardScaler()
        self.fitted = False

    def impute_and_scale(self, data):
        """
        Impute and scale non-categorical features.
        Parameters:
            data (pandas df): clean encoded data to be imputed and scaled
        Returns:
            encoded_imputed_scaled_data (pandas df): encoded, imputed and scaled data
        """
        imputed_scaled_data = None

        # if imputer and scaler not fitted, fit and transform
        if not self.fitted:
            imputed_data = pd.DataFrame(self.imputer.fit_transform(data[self.non_categorical_features]),
                                        columns=data[self.non_categorical_features].columns)
            imputed_scaled_data = pd.DataFrame(self.scaler.fit_transform(imputed_data), columns=imputed_data.columns)
            self.fitted = True

        # Otherwise, transform using fitted imputer and scaler
        else:
            imputed_data = pd.DataFrame(self.imputer.transform(data[self.non_categorical_features]),
                                        columns=data[self.non_categorical_features].columns)
            imputed_scaled_data = pd.DataFrame(self.scaler.transform(imputed_data), columns=imputed_data.columns)

        # Combine encoded categorical features with imputed scaled non-categorical features
        all_columns = set(data.columns)
        encoded_features = list(all_columns - set(self.non_categorical_features))
        encoded_imputed_scaled_data = pd.concat([imputed_scaled_data, data[encoded_features]], axis=1)
        return encoded_imputed_scaled_data


if __name__ == "__main__":
    # read_data with the correct key to retrieve the filepath
    data_train = read_data()

    # Call
    data_train_clean = fix_symbol_and_data_type(data_train)
    print(data_train_clean.head())  # Print the first few rows of the dataframe
    # print(test.head())
