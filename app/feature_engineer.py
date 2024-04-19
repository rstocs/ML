import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess_data as preprocess
from load_data import read_data
from sklearn.linear_model import LogisticRegression
from split_training_data import create_train_val_test_data as cd
    
def visualize_correlation(data):
    sns.set(style='white')
    corr = data.corr()
    plt.figure(figsize=(12,12))
    sns.set(font_scale=1)
    sns.heatmap(data=corr,
                center=0,
                cmap=sns.diverging_palette(220, 10, as_cmap=True), 
                square=True, linewidth=0.5)
    plt.show()

## perform feature engineer using lasso
def initial_feature_selection(data):
    """
    Select a list of features for training.
    Parameters:
        data (pandas df): imputed scaled encoded training data for feature selection.
    Returns:
        list of str: A list that holds the names of the selected features.
    """
    exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    exploratory_LR.fit(data.drop(columns=['y']), data['y'])
    exploratory_results = pd.DataFrame(data.drop(columns=['y']).columns).rename(columns={0:'name'})
    exploratory_results['coefs'] = exploratory_LR.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs']**2

    ## Get num of top features from config
    with open("../config/features.json") as file:
        feature_config = json.load(file)
    num_of_features = feature_config["NUM_OF_FEATURES_SELECTED"]

    var_reduced = exploratory_results.nlargest(num_of_features,'coefs_squared')
    return var_reduced['name'].to_list()

## tests, show imputer statistics and variance
if __name__ == "__main__":
    # Read training data
    data = read_data()
    print("data is: ", data.shape)

    # create train, val and test data
    train, val, test = cd(data)

    # Get categorical features list
    with open("../config/features.json") as file:
        feature_config = json.load(file)
    categorical_features = feature_config["CATEGORICAL_FEATURES"]

    # preprocess the training data: impute missing data, standard scaling and encoding categorical features
    preprocessed_data = preprocess.PreprocessData(categorical_features=[], non_categorical_features=[], imputer=None)
    imputed_scaled_encoded = preprocessed_data.prepare_data(train)

    # visualize preprocessed training data
    visualize_correlation(imputed_scaled_encoded)

    # Select the top features. The number of features is specified by the config
    print("selected feature is", initial_feature_selection(imputed_scaled_encoded))
    print("\nDone")
