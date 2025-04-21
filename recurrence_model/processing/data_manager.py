import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from recurrence_model import __version__ as _version
from recurrence_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


## Pre-pipeline Preparation

def rename_columns(*, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names in the dataframe.
    
    Args:
        dataframe: The input dataframe
        
    Returns:
        DataFrame with standardized column names
    """
    # Create a copy to avoid modifying the original
    df = dataframe.copy()
    
    # Column mapping dictionary - both variants of the misspelled column
    column_mapping = {
        "Hx Radiothreapy": "HxRadiotherapy"
    }
    
    # Rename columns that exist in the dataframe
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Load dataset from file.
    
    Args:
        file_name: Name of the CSV file to load
        
    Returns:
        Pandas DataFrame containing the data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If there's an issue parsing the CSV
    """
    try:
        file_path = DATASET_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        dataframe = pd.read_csv(file_path)
        
        
        if dataframe.empty:
            raise ValueError("The loaded dataset is empty")
            
        return dataframe
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error loading dataset: {e}")
        raise

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous saved models. 
    This ensures that when the package is published, there is only one trained model that 
    can be called, and we know exactly how it was built.
    """
    try:
        # Prepare versioned save file name
        save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
        save_path = TRAINED_MODEL_DIR / save_file_name
        
        # Create directory if it doesn't exist
        TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        remove_old_pipelines(files_to_keep=[save_file_name])
        joblib.dump(pipeline_to_persist, save_path)
        print("Model/pipeline saved successfully.")
    except Exception as e:
        print(f"Error saving pipeline: {e}")
        raise

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline.
    
    Args:
        file_name: Name of the pipeline file to load
        
    Returns:
        Trained scikit-learn Pipeline object
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    try:
        file_path = TRAINED_MODEL_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at {file_path}")
            
        trained_model = joblib.load(filename=file_path)
        
        return trained_model
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading model: {e}")
        raise

def check_model_version_compatibility(model_version: str) -> bool:
    """Check if the loaded model version is compatible with the current code version.
    
    Args:
        model_version: The version string of the loaded model
        
    Returns:
        Boolean indicating compatibility
    """
    # Simple version check - in real applications, use more sophisticated versioning
    return model_version == _version

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()