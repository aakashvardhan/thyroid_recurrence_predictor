import sys
from pathlib import Path

# Add root directory to path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from recurrence_model.config.core import config
from recurrence_model.processing.data_manager import load_dataset, rename_columns


@pytest.fixture(scope="session")
def sample_input_data():
    """Load sample data for testing."""
    # Load the dataset
    data = load_dataset(file_name=config.app_config.training_data_file)
    
    # Rename columns to handle any inconsistencies
    data = rename_columns(dataframe=data)
    
    # Validate inputs minimally - ensuring required columns exist
    required_columns = config.ml_config.features + [config.ml_config.target]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle extreme values and missing data
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_columns:
        # Replace inf values with NaN
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())
    
    return data


@pytest.fixture(scope="session")
def test_data(sample_input_data):
    """Return X_test and y_test data for model evaluation."""
    # Extract features and target
    X = sample_input_data[config.ml_config.features]
    y = sample_input_data[config.ml_config.target]
    
    # Map categorical target to binary
    y = y.map(config.ml_config.recurrence_map)
    
    # Split data using the same random state as the training pipeline
    _, X_test, _, y_test = train_test_split(
        X, 
        y,
        test_size=config.ml_config.test_size,
        random_state=config.ml_config.random_state
    )
    
    return X_test, y_test
