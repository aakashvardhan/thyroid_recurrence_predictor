import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from recurrence_model import __version__ as model_version
from recurrence_model.predict import make_prediction


from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.post("/recurrence", response_model=schemas.Model)
def recurrence() -> dict:
    """
    Root Get
    """
    recurrence = schemas.Model(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )
    return recurrence.model_dump()


example_input = {
    "inputs": [
        {
            "Age": 77,
            "Gender": "M",
            "HxRadiotherapy": "Yes",
            "Adenopathy": "No",
            "Pathology": "Micropapillary",
            "Focality": "Uni-Focal",
            "Risk": "Intermediate",
            "T": "T1a",
            "N": "N0",
            "M": "M0",
            "Stage": "I",
            "Response": "Indeterminate",
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Recurrence prediction with the recurrence_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    try:
        results = make_prediction(input_data=input_df.replace({np.nan: None}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return results
