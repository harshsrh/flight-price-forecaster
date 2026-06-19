import numpy as np
import pandas as pd
import datetime

def encode_request(request, encoders):
    """Convert the incoming request into model-ready features."""
    def safe_encode(col, value, default=0):
        if encoders and col in encoders:
            le = encoders[col]
            if value in le.classes_:
                return int(le.transform([value])[0])
        return default

    route_str = f"{request.source_city}_{request.destination_city}"
    stops_map = {"zero": 0, "one": 1, "two_or_more": 2}
    time_map = {"Early_Morning": 0, "Morning": 1, "Afternoon": 2,
                "Evening": 3, "Night": 4, "Late_Night": 5}

    features = {
        "airline_enc": safe_encode("airline", request.airline),
        "source_city_enc": safe_encode("source_city", request.source_city),
        "destination_city_enc": safe_encode("destination_city", request.destination_city),
        "route_enc": safe_encode("route", route_str),
        "stops_num": stops_map.get(request.stops, 0),
        "dep_time_num": time_map.get(request.departure_time, 1),
        "arr_time_num": time_map.get(request.arrival_time, 3),
        "duration_total_mins": request.duration_hours * 60,
        "days_left": request.days_left,
        "is_last_minute": int(request.days_left <= 3),
        "is_advance_booking": int(request.days_left >= 30),
        "is_economy": int(request.travel_class == "Economy"),
        "booking_window_enc": 0,
    }
    return features, route_str

def predict_with_rf(features, registry):
    X = pd.DataFrame([features])
    pred = registry.random_forest.predict(X)[0]
    return float(pred)

def predict_with_prophet(route_str, days_left, registry):
    """Use Prophet model if one exists for this exact route."""
    model = registry.prophet_models.get(route_str)
    if model is None:
        return None, None, None

    target_date = datetime.date.today() + datetime.timedelta(days=int(days_left))
    future = pd.DataFrame({
        "ds": [pd.to_datetime(target_date)],
        "stops_num": [0],
        "is_economy": [1]
    })
    forecast = model.predict(future)
    yhat = float(forecast["yhat"].iloc[0])
    yhat_lower = float(forecast["yhat_lower"].iloc[0])
    yhat_upper = float(forecast["yhat_upper"].iloc[0])
    return yhat, yhat_lower, yhat_upper

def determine_trend(route_str, days_left, registry, current_price):
    """Compare price now vs price 7 days from now to suggest buy/wait."""
    model = registry.prophet_models.get(route_str)
    if model is None:
        return "stable", "buy_now"

    future_date = datetime.date.today() + datetime.timedelta(days=int(days_left) - 7)
    future = pd.DataFrame({
        "ds": [pd.to_datetime(future_date)],
        "stops_num": [0],
        "is_economy": [1]
    })
    forecast = model.predict(future)
    future_price = float(forecast["yhat"].iloc[0])

    diff_pct = (future_price - current_price) / current_price * 100

    if diff_pct < -3:
        return "falling", "wait"
    elif diff_pct > 3:
        return "rising", "buy_now"
    else:
        return "stable", "buy_now"

def make_prediction(request, registry):
    features, route_str = encode_request(request, registry.label_encoders)

    # Try Prophet first (route-specific, has confidence intervals)
    prophet_pred, lower, upper = predict_with_prophet(
        route_str, request.days_left, registry
    )

    if prophet_pred is not None:
        price = prophet_pred
        model_used = "Prophet"
        conf_low, conf_high = lower, upper
    else:
        # Fallback to Random Forest (works for any route)
        price = predict_with_rf(features, registry)
        model_used = "RandomForest"
        conf_low, conf_high = price * 0.9, price * 1.1

    trend, recommendation = determine_trend(route_str, request.days_left, registry, price)

    return {
        "predicted_price": round(price, 2),
        "model_used": model_used,
        "confidence_low": round(conf_low, 2) if conf_low else None,
        "confidence_high": round(conf_high, 2) if conf_high else None,
        "trend": trend,
        "recommendation": recommendation,
        "route": route_str
    }