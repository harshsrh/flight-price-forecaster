from frontend.api_client import predict_price
import pandas as pd
import os



def build_forecast_curve(source_city, destination_city, airline,
                          stops="zero", travel_class="Economy", max_days=30):
    """Calls the API once per day-out to build a price curve."""
    days_range = list(range(1, max_days + 1))
    prices = []

    for d in days_range:
        result, error = predict_price(
            source_city, destination_city, airline, d,
            stops=stops, travel_class=travel_class
        )
        if result:
            prices.append(result["predicted_price"])
        else:
            prices.append(None)

    return days_range, prices



def load_model_metrics():
    """Loads your saved model comparison results for display in the dashboard."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "final_comparison.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        best = df.sort_values("rmse").iloc[0]
        return {
            "rmse": best["rmse"],
            "mae": best["mae"],
            "directional_acc": best["directional_acc"]
        }
    return {"rmse": None, "mae": None, "directional_acc": None}