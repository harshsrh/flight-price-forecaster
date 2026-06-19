from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.schemas import PredictionRequest, PredictionResponse, HistoryResponse
from api.model_loader import registry
from api.predict import make_prediction
from api.database import init_db, log_prediction, get_price_history

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models once
    init_db()
    registry.load_all()
    yield
    # Shutdown: nothing needed

app = FastAPI(
    title="Flight Price Forecasting API",
    description="Predicts flight prices and recommends buy-now vs wait-and-save",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Flight Price Forecasting API",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": registry.loaded,
        "prophet_routes": list(registry.prophet_models.keys()),
        "lstm_routes": list(registry.lstm_models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        result = make_prediction(request, registry)

        log_prediction(
            route=result["route"],
            airline=request.airline,
            days_left=request.days_left,
            stops=request.stops,
            travel_class=request.travel_class,
            predicted_price=result["predicted_price"],
            model_used=result["model_used"],
            conf_low=result["confidence_low"],
            conf_high=result["confidence_high"]
        )

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=HistoryResponse)
def history(source_city: str, destination_city: str, limit: int = 100):
    route = f"{source_city}_{destination_city}"
    points = get_price_history(route, limit=limit)

    if not points:
        raise HTTPException(status_code=404, detail=f"No history found for route {route}")

    return HistoryResponse(route=route, points=points)

@app.get("/routes")
def available_routes():
    """List all routes that have a dedicated Prophet/LSTM model."""
    return {
        "prophet_routes": list(registry.prophet_models.keys()),
        "lstm_routes": list(registry.lstm_models.keys())
    }