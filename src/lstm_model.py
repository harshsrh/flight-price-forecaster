import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

model_dir="models"
SEQ_LEN=14
os.makedirs(model_dir, exist_ok=True)

def build_sequences(prices, seq_len=SEQ_LEN):
    """Convert a flat price array into (X, y) sequences for LSTM."""
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i + seq_len])
        y.append(prices[i + seq_len])
    return np.array(X), np.array(y)

def prepare_lstm_data(df, route=None):
    if route:
        df = df[df['route'] == route].copy()

    # Sort by days_left descending = chronological order
    df = df.sort_values('days_left', ascending=False)
    prices = df['price'].values.reshape(-1, 1)

    # Scale to [0, 1] — critical for LSTM convergence
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices).flatten()

    # Build sequences
    X, y = build_sequences(prices_scaled, SEQ_LEN)

    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Time-based split (never random for time series)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler


def build_lstm_model(seq_len=SEQ_LEN):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    return model

def train_lstm(X_train, y_train, route_name="all_routes"):
    print(f"\nTraining LSTM for: {route_name}")
    print(f"Training samples: {len(X_train)}")

    model = build_lstm_model()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,            # stop if no improvement for 10 epochs
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"{model_dir}/lstm_{route_name.replace(' ','_')}.keras",
            save_best_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Trained for {len(history.history['loss'])} epochs")
    return model, history

def evaluate_lstm(model, X_test, y_test, scaler):
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform back to real prices
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # Directional accuracy
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    da = (np.sign(true_diff) == np.sign(pred_diff)).mean() * 100

    print(f"\n  RMSE             : {rmse:.2f}")
    print(f"  MAE              : {mae:.2f}")
    print(f"  Directional Acc. : {da:.1f}%")

    return {"rmse": rmse, "mae": mae, "directional_acc": da,
            "y_true": y_true, "y_pred": y_pred}

def plot_predictions(y_true, y_pred, route_name):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label='Actual Price', linewidth=1.5)
    plt.plot(y_pred, label='LSTM Predicted', linewidth=1.5, linestyle='--')
    plt.title(f'LSTM Predictions vs Actual — {route_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/lstm_predictions_{route_name.replace(' ','_')}.png", dpi=150)
    plt.show()
    print(f"Saved plot → results/lstm_predictions_{route_name}.png")

def run_lstm_pipeline():
    df = pd.read_csv("data/processed/clean.csv")
    top_routes = df['route'].value_counts().head(3).index.tolist()

    all_results = []

    for route in top_routes:
        df_route = df[df['route'] == route]

        if len(df_route) < SEQ_LEN * 5:
            print(f"Skipping {route} — insufficient data")
            continue

        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, route=route)

        if len(X_train) < 50:
            print(f"Skipping {route} — too few sequences ({len(X_train)})")
            continue

        model, history = train_lstm(X_train, y_train, route_name=route)

        # Save scaler — needed for inference in the API
        joblib.dump(scaler, f"{model_dir}/lstm_scaler_{route.replace(' ','_')}.pkl")

        metrics = evaluate_lstm(model, X_test, y_test, scaler)
        plot_predictions(metrics['y_true'], metrics['y_pred'], route)

        all_results.append({"route": route,
                            "rmse": metrics['rmse'],
                            "mae": metrics['mae'],
                            "directional_acc": metrics['directional_acc']})

    # Final comparison table
    print("\n" + "="*55)
    print("  FINAL MODEL COMPARISON")
    print("="*55)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    results_df.to_csv("results/lstm_results.csv", index=False)
    print("\nSaved → results/lstm_results.csv")

    return all_results

if __name__ == "__main__":
    run_lstm_pipeline()
