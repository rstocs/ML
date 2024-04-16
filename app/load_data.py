import json
import pandas as pd

# write unit test to:
# 1. check counter of 1's and 0's.
# 2. check object dtype
# 3. int64 dtype
# 4. float64 dtypes
# 5. investigate object columns
def read_data(filepath):
    with open("SETTINGS.json") as file:
        settings = json.load(file)
    return pd.read_csv(settings[filepath])

if __name__ == "__main__":
    # filepath key that exist in SETTINGS.json
    filepath_key = 'TRAIN_DATA_PATH'
    # Call read_data with the correct key to retrieve the filepath
    data_frame = read_data(filepath_key)
    print(data_frame.head())  # Print the first few rows of the dataframe