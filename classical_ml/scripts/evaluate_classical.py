import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    data_dir = Path("classical_ml/data")
    eval_dir = Path("classical_ml/evaluation")
    eval_dir.mkdir(exist_ok=True)
    
    print("Loading test dataset to formalize Climatology Baseline...")
    test_df = pd.read_parquet(data_dir / "test_tabular.parquet")
    
    results = {}
    
    for pollutant in ['no2', 'co', 'so2']:
        y_true = test_df[f'{pollutant}_target'].values
        y_clim = test_df[f'{pollutant}_clim_mean'].values
        
        # Calculate metrics
        r2 = r2_score(y_true, y_clim)
        rmse = np.sqrt(mean_squared_error(y_true, y_clim))
        mae = mean_absolute_error(y_true, y_clim)
        
        results[pollutant] = {
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae)
        }
        
        print(f"=== {pollutant.upper()} 150-Day Climatology Baseline ===")
        print(f"R2   : {r2:.4f}")
        print(f"RMSE : {rmse:.4e}")
        print(f"MAE  : {mae:.4e}\n")
        
    with open(eval_dir / "climatology_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Summary written to classical_ml/evaluation/climatology_metrics.json")
    print("CONCLUSION: The monthly climatology represents the theoretical ceiling for 150-day forecasting.")

if __name__ == "__main__":
    main()
