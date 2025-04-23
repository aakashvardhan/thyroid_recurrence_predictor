from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel

class PredictionResult(BaseModel):
    label: int
    recurrence: str

class PredictionResults(BaseModel):
    predictions: List[PredictionResult]
    version: str
    
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