import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def train_and_evaluate(target_name, train_df, test_df, features):
    print(f"\n{'='*40}")
    print(f"Maximizing {target_name.upper()} Forecasting...")
    print(f"{'='*40}")
    
    # We predict the raw target directly, allowing the tree to learn non-linear spatial-climatology
    X = train_df[features]
    y = train_df[f'{target_name}_target']
    
    # We split 10% of train data for early stopping to completely prevent overfitting
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    X_test = test_df[features]
    y_test = test_df[f'{target_name}_target']
    
    print("Training Highly-Tuned LightGBM Regressor with Early Stopping...")
    
    # Hyperparameters tuned for maximum generalization on spatial-seasonal structures
    model = lgb.LGBMRegressor(
        n_estimators=1000,         # High number of trees
        learning_rate=0.01,        # Very slow learning rate for smooth convergence
        max_depth=9,               # Deeper trees to capture (x, y, DOY) interactions
        num_leaves=128,            # Allow complex spatial boundaries
        subsample=0.7,             # Bagging to reduce variance
        colsample_bytree=0.7,      # Feature bagging
        min_child_samples=50,      # Prevent overfitting to single outlier pixels
        random_state=42,
        n_jobs=-1
    )
    
    # Use early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    print(f"Model converged at iteration {model.best_iteration_}")
    
    # Evaluate
    pred_test = model.predict(X_test)
    
    r2_model = r2_score(y_test, pred_test)
    rmse_model = np.sqrt(mean_squared_error(y_test, pred_test))
    
    # Compare to simple monthly climatology
    y_clim = test_df[f'{target_name}_clim_mean']
    r2_clim = r2_score(y_test, y_clim)
    
    print(f"--- Results on 2024-2025 Holdout ---")
    print(f"Monthly Climatology R² : {r2_clim:.4f}")
    print(f"Max-Optimized ML R²    : {r2_model:.4f}")
    
    if r2_model > r2_clim:
        print(f"SUCCESS! We broke the climatology ceiling by {(r2_model - r2_clim):.4f} points!")
    else:
        print(f"Still bounded by climatology ceiling. The ML model cannot extract more signal.")

def main():
    data_dir = Path("classical_ml/data")
    
    print("Loading datasets...")
    train_df = pd.read_parquet(data_dir / "train_tabular.parquet")
    test_df = pd.read_parquet(data_dir / "test_tabular.parquet")
    
    # We use all available structural features
    features = ['x_norm', 'y_norm', 'sin_doy', 'cos_doy', 'ndvi', 'ndbi', 'ndmi', 'so2_dist']
    
    train_and_evaluate('no2', train_df, test_df, features)
    train_and_evaluate('co', train_df, test_df, features)

if __name__ == "__main__":
    main()
