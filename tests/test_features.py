import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from recurrence_model.processing.features import OrdinalCategoricalMapper
from recurrence_model.config.core import config


def test_ordinal_categorical_mapper(sample_input_data):
    # Setup
    test_df = sample_input_data.copy()
    risk_column = config.ml_config.risk_var
    risk_mapping = config.ml_config.risk_map

    # Create mapper with risk mapping
    mapper = OrdinalCategoricalMapper(
        mappings={risk_column: risk_mapping},
        handle_unknown="use_default",
        default_value=0,
    )

    # Transform data
    transformed_df = mapper.fit_transform(test_df)

    # Assert the transformation worked correctly
    assert risk_column in transformed_df.columns
    assert transformed_df[risk_column].isin(list(risk_mapping.values())).all()
    assert not transformed_df[risk_column].isna().any()

    # Test with multiple mappings
    multi_mapper = OrdinalCategoricalMapper(
        mappings={
            config.ml_config.risk_var: config.ml_config.risk_map,
            config.ml_config.stage_var: config.ml_config.stage_map,
        },
        handle_unknown="use_default",
        default_value=0,
    )

    multi_transformed = multi_mapper.fit_transform(test_df)
    assert (
        multi_transformed[config.ml_config.risk_var]
        .isin(list(config.ml_config.risk_map.values()))
        .all()
    )
    assert (
        multi_transformed[config.ml_config.stage_var]
        .isin(list(config.ml_config.stage_map.values()))
        .all()
    )


def test_one_hot_encoder(sample_input_data):
    # Setup
    test_df = sample_input_data.copy()
    categorical_feature = config.ml_config.gender_var

    # Create a test dataframe with just one categorical column
    X = test_df[[categorical_feature]]

    # Initialize encoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Fit and transform
    encoded_array = encoder.fit_transform(X)

    # Check shape matches expected number of categories
    unique_values = X[categorical_feature].nunique()
    assert encoded_array.shape[1] == unique_values

    # Test for multiple categorical columns
    categorical_columns = [config.ml_config.gender_var, config.ml_config.adenopathy_var]

    X_multi = test_df[categorical_columns]
    encoder_multi = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_multi = encoder_multi.fit_transform(X_multi)

    # Check total number of columns in encoded result
    expected_cols = sum(test_df[col].nunique() for col in categorical_columns)
    assert encoded_multi.shape[1] == expected_cols

    # Verify one-hot property: each row should sum to the number of features
    assert np.all(np.sum(encoded_multi, axis=1) == len(categorical_columns))
