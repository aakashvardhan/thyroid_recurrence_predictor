import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging
from datetime import datetime
import numpy as np

from recurrence_model.config.core import config, TRAINED_MODEL_DIR
from recurrence_model.processing.data_manager import load_dataset, save_pipeline, rename_columns
from recurrence_model.pipeline import recurrence_pipe
from recurrence_model import __version__ as _version

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Validate the input data.

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
    required_columns = config.ml_config.features + [config.ml_config.target]
    missing_columns = [col for col in required_columns if col not in data_copy.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for missing values in target column
    if data_copy[config.ml_config.target].isna().any():
        raise ValueError("Missing values found in target column")

    # Handle extreme values without inplace=True
    numeric_columns = data_copy.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_columns:
        # Replace inf values with NaN without inplace
        data_copy[col] = data_copy[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median without inplace
        if data_copy[col].isna().any():
            data_copy[col] = data_copy[col].fillna(data_copy[col].median())

    return data_copy


def save_metrics(*, metrics: dict, filename: str = "metrics.json") -> None:
    """Save model metrics to a file.

    Args:
        metrics: Dictionary of metrics to save
        filename: Name of the file to save metrics to
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_with_timestamp = {
        "timestamp": timestamp,
        "version": _version,
        "metrics": metrics,
    }

    # Create directory if it doesn't exist
    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = TRAINED_MODEL_DIR / filename

    with open(metrics_file, "w") as f:
        json.dump(metrics_with_timestamp, f, indent=4)

    logger.info(f"Model metrics saved to {metrics_file}")


def run_training() -> None:
    """Train the model."""
    try:
        # Read training data
        logger.info("Loading dataset")
        data = load_dataset(file_name=config.app_config.training_data_file)
        
        # Rename columns to match DataInputSchema
        logger.info("Renaming columns to match schema")
        data = rename_columns(dataframe=data)

        # Validate inputs after renaming columns
        logger.info("Validating data")
        validated_data = validate_inputs(data)

        # Divide train and test
        X_train, X_test, y_train, y_test = train_test_split(
            validated_data[config.ml_config.features],
            validated_data[config.ml_config.target],
            test_size=config.ml_config.test_size,
            random_state=config.ml_config.random_state,
        )

        y_train = y_train.map(config.ml_config.recurrence_map)
        y_test = y_test.map(config.ml_config.recurrence_map)

        # Fit the pipeline
        logger.info("Training model")
        recurrence_pipe.fit(X_train, y_train)
        y_pred = recurrence_pipe.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Print metrics
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.capitalize()}: {metric_value:.4f}")

        # Save metrics
        save_metrics(metrics=metrics)

        # Save the pipeline
        logger.info("Saving model pipeline")
        save_pipeline(pipeline_to_persist=recurrence_pipe)

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    run_training()
