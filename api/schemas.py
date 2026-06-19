from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class TravelClass(str, Enum):
    economy = "Economy"
    business = "Business"

class StopType(str, Enum):
    zero = "zero"
    one = "one"
    two_or_more = "two_or_more"

class PredictionRequest(BaseModel):
    source_city: str = Field(..., example="Delhi")
    destination_city: str = Field(..., example="Mumbai")
    airline: str = Field(..., example="Indigo")
    days_left: int = Field(..., ge=0, le=60, example=15)
    stops: StopType = Field(default=StopType.zero)
    travel_class: TravelClass = Field(default=TravelClass.economy)
    departure_time: str = Field(default="Morning")
    arrival_time: str = Field(default="Evening")
    duration_hours: float = Field(default=2.5)

class PredictionResponse(BaseModel):
    predicted_price: float
    model_used: str
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    trend: str                       # "rising", "falling", "stable"
    recommendation: str              # "buy_now" or "wait"
    route: str

class HistoryPoint(BaseModel):
    days_left: int
    price: float
    airline: Optional[str] = None

class HistoryResponse(BaseModel):
    route: str
    points: List[HistoryPoint]

class ModelStats(BaseModel):
    model_name: str
    rmse: float
    mae: float
    directional_accuracy: float