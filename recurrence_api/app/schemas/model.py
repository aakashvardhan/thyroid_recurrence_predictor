from pydantic import BaseModel

class Model(BaseModel):
    name: str
    api_version: str
    model_version: str