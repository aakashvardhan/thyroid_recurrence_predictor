import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from recurrence_model.config.core import config
from recurrence_model.processing.features import (
    CustomOneHotEncoder,
    OrdinalCategoricalMapper,
)

# Define the pipeline for thyroid recurrence prediction
recurrence_pipe = Pipeline([
    # Ordinal mapping for categorical variables
    ("risk_mapper", OrdinalCategoricalMapper(
        mappings={config.ml_config.risk_var: config.ml_config.risk_map},
        handle_unknown='use_default',
        default_value=0
    )),
    ("t_stage_mapper", OrdinalCategoricalMapper(
        mappings={config.ml_config.t_stage_var: config.ml_config.t_stage_map},
        handle_unknown='use_default',
        default_value=0
    )),
    ("n_stage_mapper", OrdinalCategoricalMapper(
        mappings={config.ml_config.n_stage_var: config.ml_config.n_stage_map},
        handle_unknown='use_default',
        default_value=0
    )),
    ("m_stage_mapper", OrdinalCategoricalMapper(
        mappings={config.ml_config.m_stage_var: config.ml_config.m_stage_map},
        handle_unknown='use_default',
        default_value=0
    )),
    ("stage_mapper", OrdinalCategoricalMapper(
        mappings={config.ml_config.stage_var: config.ml_config.stage_map},
        handle_unknown='use_default',
        default_value=0
    )),
    
    # One-hot encoding for categorical variables
    ("gender_encoder", CustomOneHotEncoder(columns=config.ml_config.gender_var)),
    ("hxradiotherapy_encoder", CustomOneHotEncoder(columns=config.ml_config.hxradiotherapy_var)),
    ("adenopathy_encoder", CustomOneHotEncoder(columns=config.ml_config.adenopathy_var)),
    ("pathology_encoder", CustomOneHotEncoder(columns=config.ml_config.pathology_var)),
    ("focality_encoder", CustomOneHotEncoder(columns=config.ml_config.focality_var)),
    ("response_encoder", CustomOneHotEncoder(columns=config.ml_config.response_var)),
    
    # Feature scaling
    ("scaler", StandardScaler()),
    
    # Model
    ("logistic_regression", LogisticRegression(
        C=config.ml_config.C,
        max_iter=config.ml_config.max_iter,
        tol=config.ml_config.tol,
        random_state=config.ml_config.random_state,
    )),
])

