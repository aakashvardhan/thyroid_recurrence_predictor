from typing import Dict, List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class OrdinalCategoricalMapper(BaseEstimator, TransformerMixin):
    """
    Transformer for mapping categorical variables to ordinal values using predefined mappings.
    """
    def __init__(self, mappings=None, handle_unknown='error', default_value=None):
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
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Found values not in mapping for column {column}: {missing}")
                    elif self.handle_unknown == 'use_default':
                        X_copy.loc[X_copy[column].isna(), column] = self.default_value
        
        return X_copy

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer for one-hot encoding categorical variables.
    Ensures consistent columns between fit and transform even when categories are missing.
    """
    def __init__(self, columns=None, drop_first=False, handle_unknown='ignore'):
        self.columns = columns if isinstance(columns, list) else [columns]
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.categories_ = {}
        
    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.columns:
            if col in X.columns:
                self.categories_[col] = list(X[col].unique())
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in self.columns:
            if col in X_copy.columns:
                # Check for unknown categories
                if self.handle_unknown == 'error':
                    unknown = set(X_copy[col].unique()) - set(self.categories_[col])
                    if unknown:
                        raise ValueError(f"Found unknown categories in column {col}: {unknown}")
                
                # Get all categories from fit
                categories = self.categories_[col]
                
                # Create columns for all categories seen during fit
                all_dummies = pd.DataFrame(0, index=X_copy.index, 
                                          columns=[f"{col}_{cat}" for cat in categories])
                
                # Create actual one-hot encoded columns from current data
                current_dummies = pd.get_dummies(X_copy[col], prefix=col)
                
                # Update the all_dummies with values from current_dummies
                for dummy_col in current_dummies.columns:
                    if dummy_col in all_dummies.columns:
                        all_dummies[dummy_col] = current_dummies[dummy_col]
                
                # Apply drop_first if needed
                if self.drop_first and len(categories) > 1:
                    first_col = f"{col}_{categories[0]}"
                    if first_col in all_dummies.columns:
                        all_dummies = all_dummies.drop(columns=[first_col])
                
                # Drop the original column and add the dummies
                X_copy = pd.concat([X_copy.drop(col, axis=1), all_dummies], axis=1)
        
        return X_copy