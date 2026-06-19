import sqlite3
import os
from datetime import datetime

db_path="flight_prices.db"

def get_connection():
    conn=sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory=sqlite3.Row
    return conn

def init_db():
    conn=get_connection()
    cursor=conn.cursor()

    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route TEXT NOT NULL,
            airline TEXT,
            days_left INTEGER,
            stops TEXT,
            travel_class TEXT,
            predicted_price REAL NOT NULL,
            model_used TEXT NOT NULL,
            confidence_low REAL,
            confidence_high REAL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route TEXT NOT NULL,
            days_left INTEGER NOT NULL,
            airline TEXT,
            price REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized:", db_path)

def log_prediction(route, airline, days_left, stops, travel_class,
                    predicted_price, model_used, conf_low=None, conf_high=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (route, airline, days_left, stops, travel_class,
         predicted_price, model_used, confidence_low, confidence_high, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (route, airline, days_left, stops, travel_class,
          predicted_price, model_used, conf_low, conf_high,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_price_history(route, limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT days_left, price, airline FROM price_history
        WHERE route = ? ORDER BY days_left DESC LIMIT ?
    """, (route, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def seed_price_history_from_csv(csv_path="data/processed/clean.csv"):
    """One-time seed of price_history table from your processed dataset."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM price_history")
    if cursor.fetchone()[0] > 0:
        print("price_history already seeded, skipping.")
        conn.close()
        return

    sample = df[['route', 'days_left', 'airline', 'price']].sample(
        min(5000, len(df)), random_state=42
    )
    sample.to_sql('price_history', conn, if_exists='append', index=False)
    conn.close()
    print(f"Seeded {len(sample)} rows into price_history")

if __name__=="__main__":
    init_db()
