# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import recurrence_model

PACKAGE_ROOT = Path(recurrence_model.__file__).resolve().parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """Application-level config."""
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str
    

class ModelConfig(BaseModel):
    """All configuration relevant to model training and feature engineering."""
    target: str
    features: List[str]
    
    age_var: str
    gender_var: str
    hxradiotherapy_var: str
    adenopathy_var: str
    pathology_var: str
    focality_var: str
    risk_var: str
    t_stage_var: str
    n_stage_var: str
    m_stage_var: str
    stage_var: str
    response_var: str
    
    gender_map: Dict[str, int]
    hxradiotherapy_map: Dict[str, int]
    adenopathy_map: Dict[str, int]
    pathology_map: Dict[str, int]
    focality_map: Dict[str, int]
    risk_map: Dict[str, int]
    t_stage_map: Dict[str, int]
    n_stage_map: Dict[str, int]
    m_stage_map: Dict[str, int]
    stage_map: Dict[str, int]
    response_map: Dict[str, int]
    recurrence_map: Dict[str, int] = {
        "Yes": 1,
        "No": 0,
    }
    
    test_size: float
    random_state: int
    tol: float
    C: float
    max_iter: int

class Config(BaseModel):
    """Master config object."""
    app_config: AppConfig
    ml_config: ModelConfig
    
    
def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = CONFIG_FILE_PATH) -> YAML:
    """Parse the yaml file and return a config object."""
    
    if not cfg_path:
        cfg_path = find_config_file()
        
    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
        
    # specify the data types
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        ml_config=ModelConfig(**parsed_config.data),
    )
    
    return _config


config = create_and_validate_config()




    
    