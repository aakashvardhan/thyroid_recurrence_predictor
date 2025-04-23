from typing import Dict, List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class OrdinalCategoricalMapper(BaseEstimator, TransformerMixin):
    """
    Transformer for mapping categorical variables to ordinal values using predefined mappings.
    """

    def __init__(self, mappings=None, handle_unknown="error", default_value=None):
        self.mappings = mappings or {}
        self.handle_unknown = handle_unknown
        self.default_value = default_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        for column, mapping in self.mappings.items():
            if column in X_copy.columns:
                X_copy[column] = X_copy[column].map(mapping)

                # Handle values not in mapping
                if X_copy[column].isna().any():
                    missing = X_copy.loc[X_copy[column].isna(), column].unique()
                    if self.handle_unknown == "error":
                        raise ValueError(
                            f"Found values not in mapping for column {column}: {missing}"
                        )
                    elif self.handle_unknown == "use_default":
                        X_copy.loc[X_copy[column].isna(), column] = self.default_value

        return X_copy
