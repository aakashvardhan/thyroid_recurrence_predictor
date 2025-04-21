import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from recurrence_model.config.core import config



class DataInputSchema(BaseModel):
    Age: Optional[int] = None
    Gender: Optional[str] = None
    HxRadiotherapy: Optional[str] = None
    Adenopathy: Optional[str] = None
    Pathology: Optional[str] = None
    Focality: Optional[str] = None
    Risk: Optional[str] = None
    T: Optional[str] = None
    N: Optional[str] = None
    M: Optional[str] = None
    Stage: Optional[str] = None
    Response: Optional[str] = None
    

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
    version: str


def validate_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Validate the input data against the DataInputSchema.
    
    Args:
        input_data: Input data to validate
        
    Returns:
        Validated input data
        
    Raises:
        ValueError: If validation fails
    """
    # Make a copy to avoid warnings
    data_copy = input_data.copy()
    
    # Check all required columns exist
    required_columns = config.ml_config.features
    missing_columns = [col for col in required_columns if col not in data_copy.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle extreme values without inplace=True
    numeric_columns = data_copy.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_columns:
        # Replace inf values with NaN without inplace
        data_copy[col] = data_copy[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median without inplace
        if data_copy[col].isna().any():
            data_copy[col] = data_copy[col].fillna(data_copy[col].median())
    
    return data_copy
    
    
    
    