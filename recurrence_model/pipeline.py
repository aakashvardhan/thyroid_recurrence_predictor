import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from recurrence_model.config.core import config
from recurrence_model.processing.features import OrdinalCategoricalMapper

# Define the categorical features for one-hot encoding
categorical_features_ohe = [
    config.ml_config.gender_var,
    config.ml_config.hxradiotherapy_var,
    config.ml_config.adenopathy_var,
    config.ml_config.pathology_var,
    config.ml_config.focality_var,
    config.ml_config.response_var
]

# Define the categorical features for ordinal encoding
categorical_features_ordinal = [
    config.ml_config.risk_var,
    config.ml_config.t_stage_var,
    config.ml_config.n_stage_var,
    config.ml_config.m_stage_var,
    config.ml_config.stage_var
]

# Define numerical features (assuming 'Age' is the only one not processed by mappers)
# It will be handled by the scaler later
numerical_features = [config.ml_config.age_var]

# Create the preprocessor using ColumnTransformer
# This allows different transformations for different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal_risk', OrdinalCategoricalMapper(mappings={config.ml_config.risk_var: config.ml_config.risk_map}, handle_unknown='use_default', default_value=0), [config.ml_config.risk_var]),
        ('ordinal_t_stage', OrdinalCategoricalMapper(mappings={config.ml_config.t_stage_var: config.ml_config.t_stage_map}, handle_unknown='use_default', default_value=0), [config.ml_config.t_stage_var]),
        ('ordinal_n_stage', OrdinalCategoricalMapper(mappings={config.ml_config.n_stage_var: config.ml_config.n_stage_map}, handle_unknown='use_default', default_value=0), [config.ml_config.n_stage_var]),
        ('ordinal_m_stage', OrdinalCategoricalMapper(mappings={config.ml_config.m_stage_var: config.ml_config.m_stage_map}, handle_unknown='use_default', default_value=0), [config.ml_config.m_stage_var]),
        ('ordinal_stage', OrdinalCategoricalMapper(mappings={config.ml_config.stage_var: config.ml_config.stage_map}, handle_unknown='use_default', default_value=0), [config.ml_config.stage_var]),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_ohe)
    ],
    remainder='passthrough' # Keep other columns (like 'Age') untouched for the next step
)


# Define the pipeline for thyroid recurrence prediction
recurrence_pipe = Pipeline([
    # Preprocessing step
    ('preprocess', preprocessor),
    
    # Feature scaling - applied to all features coming out of preprocessor
    ("scaler", StandardScaler()),
    
    # Model
    ("logistic_regression", LogisticRegression(
        C=config.ml_config.C,
        max_iter=config.ml_config.max_iter,
        tol=config.ml_config.tol,
        random_state=config.ml_config.random_state,
    )),
])

