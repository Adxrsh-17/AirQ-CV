import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def main():
    data_dir = Path("classical_ml/data")
    models_dir = Path("classical_ml/models_saved")
    models_dir.mkdir(exist_ok=True)
    
    print("Loading tabular data...")
    train_df = pd.read_parquet(data_dir / "train_tabular.parquet")
    test_df = pd.read_parquet(data_dir / "test_tabular.parquet")
    
    # Target is SO2 Anomaly
    train_df['so2_anomaly'] = train_df['so2_target'] - train_df['so2_clim_mean']
    test_df['so2_anomaly'] = test_df['so2_target'] - test_df['so2_clim_mean']
    
    # We heavily feature the precomputed distance to hotspot
    features = ['so2_dist', 'sin_doy', 'cos_doy', 'x_norm', 'y_norm', 'ndbi']
    
    X_train = train_df[features]
    y_train = train_df['so2_anomaly']
    
    X_test = test_df[features]
    y_test_anomaly = test_df['so2_anomaly']
    y_test_true = test_df['so2_target']
    y_test_clim = test_df['so2_clim_mean']
    
    print("Training LightGBM Distance-Decay Model for SO2...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,       # keep it shallow to enforce smooth distance decay
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Saving model...")
    model.booster_.save_model(models_dir / 'so2_distance_lgbm.txt')
    
    print("Evaluating on Holdout Test Set (2024-2025)...")
    pred_anomaly = model.predict(X_test)
    pred_full = y_test_clim + pred_anomaly
    
    # Compute Metrics
    r2_clim = r2_score(y_test_true, y_test_clim)
    r2_model = r2_score(y_test_true, pred_full)
    
    rmse_clim = np.sqrt(mean_squared_error(y_test_true, y_test_clim))
    rmse_model = np.sqrt(mean_squared_error(y_test_true, pred_full))
    
    print(f"=== SO2 Long-Range (150-Day) Performance ===")
    print(f"Climatology Baseline R² : {r2_clim:.4f}")
    print(f"Climatology Baseline RMSE: {rmse_clim:.4e}")
    print(f"Distance-Decay RF R²    : {r2_model:.4f}")
    print(f"Distance-Decay RF RMSE  : {rmse_model:.4e}")
    
    if r2_model > r2_clim:
        print(f"Success! Model beat climatology by {(r2_model - r2_clim):.4f} R² points.")
    else:
        print(f"Warning: Model failed to beat climatology.")

if __name__ == "__main__":
    main()

