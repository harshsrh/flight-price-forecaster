import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

model_dir='models'
os.makedirs(model_dir, exist_ok=True)

def load_splits():
    x_train=np.load("data/processed/x_train.npy",allow_pickle=True)
    x_test=np.load("data/processed/x_test.npy",allow_pickle=True)
    y_train=np.load("data/processed/y_train.npy",allow_pickle=True)
    y_test=np.load("data/processed/y_test.npy",allow_pickle=True)

    return x_train, x_test, y_train, y_test

def directional_accuracy(y_true,y_pred):
    """% of times model correctly predicts price went up or down."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    correct = np.sign(true_diff) == np.sign(pred_diff)
    return correct.mean() * 100

def evaluate(name, y_test, y_pred):
    rmse=np.sqrt(mean_squared_error(y_test, y_pred))
    mae=mean_absolute_error(y_test,y_pred)
    da= directional_accuracy(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  RMSE               : {rmse:.2f}")
    print(f"  MAE                : {mae:.2f}")
    print(f"  Directional Acc.   : {da:.1f}%")
    return {"model": name, "rmse": rmse, "mae": mae, "directional_acc": da}

def train_random_forest(x_train, y_train):
    print('training random forest')
    rf=RandomForestRegressor(n_estimators=100,
                             max_depth=15,
                             min_samples_split=5,
                             n_jobs=-1,
                             random_state=42)
    rf.fit(x_train,y_train)
    joblib.dump(rf,f"{model_dir}/random_forest.pkl")
    return rf

def train_gradient_boosting(x_train, y_train):
    print("training gradient boosting")
    gb=GradientBoostingRegressor(n_estimators=200,
                                 learning_rate=0.05,
                                 max_depth=5,
                                 random_state=42)
    
    gb.fit(x_train,y_train)
    joblib.dump(gb,f"{model_dir}/gradient_boosting.pkl")
    return gb

def run_baseline():
    x_train,x_test,y_train,y_test=load_splits()
    results=[]

    rf=train_random_forest(x_train, y_train)
    rf_pred=rf.predict(x_test)
    results.append(evaluate("Random forest",y_test,rf_pred))

    gb = train_gradient_boosting(x_train, y_train)
    gb_pred = gb.predict(x_test)
    results.append(evaluate("Gradient Boosting", y_test, gb_pred))

    feature_names = [
        'airline', 'source_city', 'dest_city', 'route', 'stops',
        'dep_time', 'arr_time', 'duration_mins', 'days_left',
        'is_last_minute', 'is_advance', 'is_economy', 'booking_window'
    ]
    importances = rf.feature_importances_
    feat_imp = sorted(zip(feature_names[:len(importances)], importances),
                      key=lambda x: x[1], reverse=True)
    print("\nTop features (Random Forest):")
    for name, imp in feat_imp[:8]:
        bar = "█" * int(imp * 50)
        print(f"  {name:<20} {bar} {imp:.3f}")


    results_df=pd.DataFrame(results)
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("\nSaved → results/model_comparison.csv")

    return results

if __name__=="__main__":
    os.makedirs('results', exist_ok=True)
    run_baseline()