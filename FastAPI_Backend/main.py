from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import List,Optional
import pandas as pd
from model import recommend,output_recommended_recipes

dataset = pd.read_csv('/Users/chhayankkashyap/Diet-Recommendation-System/Data/dataset.csv', compression='gzip')

app = FastAPI()

class params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False


class PredictionIn(BaseModel):
    nutrition_input: List[float] = Field(..., min_items=9, max_items=9) 
    ingredients:list[str]=[]
    params:Optional[params]
                    
class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: List[str]  # Use List instead of list
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]  # Use List instead of list

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.post("/predict/", response_model=PredictionOut)
def update_item(prediction_input: PredictionIn):
    recommendation_dataframe = recommend(
        dataset,
        prediction_input.nutrition_input,
        prediction_input.ingredients,
        prediction_input.params.dict() if prediction_input.params else {}  # Handle optional params correctly
    )
    output = output_recommended_recipes(recommendation_dataframe)
    return {"output": output} if output is not None else {"output": None}
