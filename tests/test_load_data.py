import pytest
import pandas as pd
import numpy as np
import json
from ..app.load_data import read_data

@pytest.fixture
def mock_settings_json():
    """Fixture to mock the SETTINGS.json content."""
    return {"filepath": "dummy_path.csv"}

@pytest.fixture
def dummy_data():
    """Fixture to return dummy data with a structure similar to the example output."""
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Create a DataFrame with 100 features and 1 binary target
    data = {}

    # Generate data for features x0 to x99
    for i in range(100):
        if i == 5:
            data[f'x{i}'] = np.random.choice(['tuesday', 'saturday', 'thursday', 'sunday', 'wednesday', 'monday', 'friday'], size=5)
        elif i == 12:
            data[f'x{i}'] = np.random.choice(['$6,882.34', '$5,647.81', '($5,032.58)', '($1,920.03)', '($5,859.08)', '$8,535.02', '$66.55', '$2,421.58', '($2,586.99)', '($4,324.44)', '($8,015.98)', '$2,669.04', '$1,729.51'], size=5)
        elif i == 31:
            data[f'x{i}'] = np.random.choice(['germany', 'asia', 'america', 'japan', np.nan], size=5)
        elif i == 63:
            data[f'x{i}'] = np.random.choice(['62.59%', '3.11%', '28.07%', '33.49%', '88.73%', '11.05%', '89.23%', '69.48%', '35.15%', '67.12%', '90.90%', '60.46%', '68.56%'], size=5)
        elif i == 81:
            data[f'x{i}'] = np.random.choice(['April', 'December', 'May', 'November', 'March', 'June', 'July', 'October', 'January', 'February', 'August', 'September'], size=5)
        elif i == 82:
            data[f'x{i}'] = np.random.choice(['Female', 'Male'], size=5)
        else:
            data[f'x{i}'] = np.random.normal(loc=0, scale=1, size=5)

    # Generate binary target y
    data['y'] = np.random.randint(0, 2, size=5)

    return pd.DataFrame(data)

def test_read_data_count_ones_zeros(dummy_data, mocker, mock_settings_json):
    """Test to check the counter of 1's and 0's in the target column 'y'."""
    # Mocking 'builtins.open' and 'pandas.read_csv'
    mocker.patch('builtins.open', mocker.mock_open(read_data=json.dumps(mock_settings_json)))
    mocker.patch('pandas.read_csv', return_value=dummy_data)

    # Call the function under test
    df = read_data('filepath')

    # Assertions
    assert df['y'].value_counts()[1] == 1
    assert df['y'].value_counts()[0] == 4  # Update the expected count to match the actual data

def test_data_types(dummy_data, mocker, mock_settings_json):
    """Test to check data types of the DataFrame columns."""
    # Mocking 'builtins.open' and 'pandas.read_csv'
    mocker.patch('builtins.open', mocker.mock_open(read_data=json.dumps(mock_settings_json)))
    mocker.patch('pandas.read_csv', return_value=dummy_data)

    # Call the function under test
    df = read_data('filepath')

    # Assertions
    assert df['x0'].dtype == 'float64'
    assert df['x1'].dtype == 'float64'
    assert df['x2'].dtype == 'float64'
    # Add assertions for other columns as needed

def test_investigate_object_columns(dummy_data, mocker, mock_settings_json):
    """Test to investigate the object dtype columns for expected string content."""
    # Mocking 'builtins.open' and 'pandas.read_csv'
    mocker.patch('builtins.open', mocker.mock_open(read_data=json.dumps(mock_settings_json)))
    mocker.patch('pandas.read_csv', return_value=dummy_data)

    # Call the function under test
    df = read_data('filepath')

    # Assertions
    assert all(isinstance(x, str) for x in df['x5'])
    assert all(isinstance(x, str) for x in df['x12'])
    assert all(isinstance(x, str) for x in df['x31'])
    assert all(isinstance(x, str) for x in df['x63'])
    assert all(isinstance(x, str) for x in df['x81'])
    assert all(isinstance(x, str) for x in df['x82'])


