import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

RAW_PATH="data/raw/Clean_Dataset.csv"
processed_path="data/processed/clean.csv"
model_dir='models'

def load_data(path=RAW_PATH):
    df=pd.read_csv(path)
    print(f"loaded {df.shape[0]:,} rows,{df.shape[1]} columns")
    return df

def clean_data(df):
    df=df.copy()

    before=len(df)

    df.drop_duplicates(inplace=True)
    print(f"Removed {before -len(df)} duplicate rows")
    df=df.loc[:,~df.columns.str.contains("^Unnamed")]

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

    # Drop rows where price is null or zero
    df = df[df['price'].notna() & (df['price'] > 0)]

    #IQR outliers 
    Q1=df['price'].quantile(0.25)
    Q3=df['price'].quantile(0.75)
    IQR=Q3-Q1
    df=df[(df['price']>=Q1-1.5* IQR)&(df['price']<=Q3+1.5*IQR)]
    print(f"After cleaning : {df.shape[0]:,} rows remain")

    return df

def engineer_features(df):
    df['is_last_minute'] = (df['days_left'] <= 3).astype(int)
    df['is_advance_booking'] = (df['days_left'] >= 30).astype(int)
    df['book_window']=pd.cut(
        df['days_left'],
        bins=[0,7,14,30,60,999],
        labels=['last_minute','one_week','two_weeks','one_month','advance']
    )

    if df['duration'].dtype == object:
        df['duration_hours'] = df['duration'].str.extract(r'(\d+)h').astype(float)
        df['duration_mins'] = df['duration'].str.extract(r'(\d+)m').fillna(0).astype(float)
        df['duration_total_mins'] = df['duration_hours'] * 60 + df['duration_mins']
    else:
        df['duration_total_mins'] = df['duration'] * 60  # already in hours

    stops_map = {'zero': 0, 'one': 1, 'two_or_more': 2}
    df['stops_num'] = df['stops'].map(stops_map).fillna(1)

    time_map = {
        'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2,
        'Evening': 3, 'Night': 4, 'Late_Night': 5
    }

    df['dep_time_num'] = df['departure_time'].map(time_map).fillna(2)
    df['arr_time_num'] = df['arrival_time'].map(time_map).fillna(2)

    df['route']=df['source_city']+'_'+df['destination_city']
    df['is_ecoomy']=(df['class']=='Economy').astype(int)

    print(f"Feature engineering complete. Columns: {list(df.columns)}")
    return df

def encode_categoricals(df):
    df=df.copy()
    le=LabelEncoder()

    cat_cols=['airline', 'source_city', 'destination_city', 'route',
                'departure_time', 'arrival_time', 'stops', 'class', 'booking_window']
    
    encoders={}
    for col in cat_cols:
        if col in df.columns:
            df[col+'_enc']=le.fit(df[col]).transform(df[col])


    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(encoders, f"{model_dir}/label_encoders.pkl")
    print(f"Saved label encoders to {model_dir}/label_encoders.pkl")

    return df, encoders

def get_feature_matrix(df):
    feature_cols = [
        'airline_enc', 'source_city_enc', 'destination_city_enc',
        'route_enc', 'stops_num', 'dep_time_num', 'arr_time_num',
        'duration_total_mins', 'days_left', 'is_last_minute',
        'is_advance_booking', 'is_economy', 'booking_window_enc'
    ]
    feature_cols=[c for c in feature_cols if c in df.columns]

    x=df[feature_cols]
    y=df['price']

    return x,y

def run_pipeline():
    os.makedirs("data/processed",exist_ok=True)

    df=load_data()
    df=clean_data(df)
    df=engineer_features(df)
    df,encoders= encode_categoricals(df)

    df.to_csv(processed_path, index=False)
    print(f"\n Saved processed data to {processed_path}")

    x,y=get_feature_matrix(df)
    print(f"\n Feature matrix: {x.shape}")
    print(f" Target (price): min={y.min():.0f},max={y.max():.0f}, mean={y.mean():.0f}")

    return df,x,y

if __name__=="__main__":
    run_pipeline()
