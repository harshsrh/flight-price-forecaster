import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

model_dir='models'
os.makedirs(model_dir, exist_ok=True)

def prepare_prophet_data(df, route=None):
    if route:
        df=df[df['route']==route].copy()

    df=df.sort_values('days_left', ascending=False).reset_index(drop=True)

    import datetime
    base_date=datetime.date.today()
    df['ds'] = [base_date - datetime.timedelta(days=int(d))
                for d in df['days_left']]
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['price']

    # Add extra regressors Prophet can use
    df['stops_num'] = df['stops_num'] if 'stops_num' in df.columns else 0
    df['is_economy'] = df['is_economy'] if 'is_economy' in df.columns else 1
    return df[['ds','y', 'stops_num','is_economy']].dropna()

def train_prophet(df_prophet, route_name="all_routes"):
    print(f"\nTraining Prophet for: {route_name} ({len(df_prophet)} rows)")

    m = Prophet(
        seasonality_mode='multiplicative',   # prices multiply, not add
        changepoint_prior_scale=0.05,        # controls trend flexibility
        seasonality_prior_scale=10,
        interval_width=0.95,                 # 95% confidence interval
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # Add custom regressors
    m.add_regressor('stops_num')
    m.add_regressor('is_economy')

    # Fit — Prophet prints its own logs
    m.fit(df_prophet)

    safe_name = route_name.replace(' ', '_')
    path = f"{model_dir}/prophet_{safe_name}.pkl"
    joblib.dump(m, path)
    print(f"Saved → {path}")
    return m

def evaluate_prophet(model, df_test):
    future = df_test[['ds', 'stops_num', 'is_economy']].copy()
    forecast = model.predict(future)

    y_true = df_test['y'].values
    y_pred = forecast['yhat'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # Directional accuracy
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    da = (np.sign(true_diff) == np.sign(pred_diff)).mean() * 100

    return {"rmse": rmse, "mae": mae, "directional_acc": da, "forecast": forecast}

def run_prophet_pipeline():
    df = pd.read_csv("data/processed/clean.csv")

    # Get top 3 routes by volume — train one Prophet model per route
    top_routes = df['route'].value_counts().head(3).index.tolist()
    print(f"Training Prophet on top routes: {top_routes}")

    all_results = []

    for route in top_routes:
        df_route = prepare_prophet_data(df, route=route)

        if len(df_route) < 100:
            print(f"Skipping {route} — not enough data ({len(df_route)} rows)")
            continue

        # 80/20 split by time (crucial — never random split time series)
        split_idx = int(len(df_route) * 0.8)
        df_train = df_route.iloc[:split_idx]
        df_test  = df_route.iloc[split_idx:]

        model = train_prophet(df_train, route_name=route)
        metrics = evaluate_prophet(model, df_test)

        print(f"\nResults for {route}:")
        print(f"  RMSE             : {metrics['rmse']:.2f}")
        print(f"  MAE              : {metrics['mae']:.2f}")
        print(f"  Directional Acc. : {metrics['directional_acc']:.1f}%")

        all_results.append({"route": route, **{k: v for k, v in metrics.items()
                                                if k != 'forecast'}})

    # Plot forecast for the best route
    best = min(all_results, key=lambda x: x['rmse'])
    print(f"\nBest Prophet route: {best['route']} (RMSE: {best['rmse']:.2f})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/prophet_results.csv", index=False)
    print("Saved → results/prophet_results.csv")

    return all_results

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_prophet_pipeline()