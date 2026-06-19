import pandas as pd
import os

def compare_all_models():
    results=[]

    for fname in ["results/model_comparison.csv",
                  "results/prophet_results.csv",
                  "results/lstm_results.csv"]:
        if os.path.exists(fname):
            df=pd.read_csv(fname)
            results.append(df)

    if not results:
        print("No results files found. Run the model scripts first.")
        return
    
    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values('rmse')

    print("\n" + "="*60)
    print("  COMPLETE MODEL COMPARISON")
    print("="*60)
    print(combined[['model', 'rmse', 'mae', 'directional_acc']].to_string(index=False))

    # Highlight winner
    best = combined.iloc[0]
    print(f"\n🏆 Best model: {best.get('model', best.get('route', 'Unknown'))}")
    print(f"   RMSE: {best['rmse']:.2f} | Directional Acc: {best['directional_acc']:.1f}%")

    combined.to_csv("results/final_comparison.csv", index=False)
    print("\nSaved → results/final_comparison.csv")

if __name__ == "__main__":
    compare_all_models()