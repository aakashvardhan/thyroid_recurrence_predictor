import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import pandas as pd
import numpy as np

from recurrence_model import __version__ as _version
from recurrence_model.config.core import config
from recurrence_model.processing.data_manager import load_pipeline, rename_columns
from recurrence_model.processing.validation import validate_inputs
import logging

def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using the saved model pipeline.
    
    Args:
        input_data: Data used for prediction, can be a single instance or multiple instances
        
    Returns:
        Predictions and model version
    """
    data = pd.DataFrame(input_data, index=[0]) if isinstance(input_data, dict) else input_data
    
    # Rename columns to match DataInputSchema
    data = rename_columns(dataframe=data)
    
    # Validate inputs
    validated_data = validate_inputs(input_data=data)
    
    # Load pipeline
    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    pipeline = load_pipeline(file_name=pipeline_file_name)
    
    # Generate predictions
    results = {"predictions": None, "version": _version}
    
    predictions = pipeline.predict(validated_data)
    results["predictions"] = [
        {"label": int(prediction), 
         "recurrence": "Yes" if prediction == 1 else "No"} 
        for prediction in predictions
    ]
    
    print(f"Making predictions with model version: {_version}")
    print(f"Predictions: {results}")
    
    return results


if __name__ == "__main__":
    data_in = {
        "Age": 77,
        "Gender": "M",
        "Hx Radiothreapy": "Yes",
        "Adenopathy": "No",
        "Pathology": "Micropapillary",
        "Focality": "Uni-Focal",
        "Risk": "Intermediate",
        "T": "T1a",
        "N": "N0",
        "M": "M0",
        "Stage": "I",
        "Response": "Indeterminate"
    }
    make_prediction(input_data=data_in)
    
    