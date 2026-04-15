from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Generic, TypeVar, Any

T = TypeVar("T")

class StandardResponse(BaseModel, Generic[T]):
    status: str = Field(..., description="'success' or 'error'")
    data: Optional[T] = Field(default=None, description="Response payload")
    message: Optional[str] = Field(default=None, description="Optional error or info message")

class SearchRequest(BaseModel):
    query: str = Field(default="", example="healthy vegetarian pasta under 500 calories")
    diet_filters: List[str] = Field(default=[], example=["Vegetarian", "Gluten-Free"])
    calorie_min: float = Field(default=0, ge=0)
    calorie_max: float = Field(default=9999, ge=0)
    meal_type: Optional[str] = Field(default=None, example="dinner")
    top_k: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    


class SimilarRequest(BaseModel):
    recipe_id: int = Field(..., example=10, ge=0)
    diet_filters: List[str] = Field(default=[], example=[])
    calorie_min: float = Field(default=0, ge=0)
    calorie_max: float = Field(default=9999, ge=0)
    top_k: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    weights: tuple[float, float, float] = Field(
        default=(0.70, 0.20, 0.10),
        description="(NLP weight, cuisine weight, nutrition weight)"
    )

class IngredientsRequest(BaseModel):
    ingredients: List[str] = Field(..., example=["chicken", "garlic", "tomato"], min_length=1)
    mode: str = Field(default="flexible", description="'exact' = only use provided, 'flexible' = allow a few extras")
    max_extra: int = Field(default=5, description="Max extra ingredients allowed (flexible mode)", ge=0)
    diet_filters: List[str] = Field(default=[])
    calorie_min: float = Field(default=0, ge=0)
    calorie_max: float = Field(default=9999, ge=0)
    top_k: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

class PantrySyncRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    ingredients: List[str] = Field(default=[])
