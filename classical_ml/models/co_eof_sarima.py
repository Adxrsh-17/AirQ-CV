import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data == -np.inf] = np.nan
        data[data == np.inf] = np.nan
        return data

def main():
    base_dir = Path("data/raw_full438")
    s5p_dir = base_dir / "s5p_composites"
    models_dir = Path("classical_ml/models_saved")
    models_dir.mkdir(exist_ok=True)
    
    t_19 = pd.Timestamp('2019-01-01', tz='UTC')
    t_23 = pd.Timestamp('2023-01-01', tz='UTC')
    t_24 = pd.Timestamp('2024-01-01', tz='UTC')
    t_25 = pd.Timestamp('2025-01-01', tz='UTC')
    
    s5p_files = sorted(list(s5p_dir.glob("s5p_*.tif")))
    
    train_dates = []
    train_co = []
    
    test_dates = []
    test_co = []
    
    # We load only the CO channel (index 1)
    print("Loading S5P arrays for CO...")
    for f in s5p_files:
        dt = pd.Timestamp(f.stem.split('_')[1], tz='UTC')
        if t_19 <= dt <= t_23:
            train_dates.append(dt)
            train_co.append(load_raster(f)[1]) # CO is idx 1
        elif t_24 < dt <= t_25:
            test_dates.append(dt)
            test_co.append(load_raster(f)[1])
            
    train_co = np.stack(train_co, axis=0) # (T, H, W)
    test_co = np.stack(test_co, axis=0)
    
    T_train, H, W = train_co.shape
    T_test = test_co.shape[0]
    
    # Flatten spatial
    train_co_flat = train_co.reshape(T_train, -1)
    test_co_flat = test_co.reshape(T_test, -1)
    
    # Handle NaNs via spatial mean for PCA
    train_co_clean = np.where(np.isnan(train_co_flat), np.nanmean(train_co_flat, axis=1, keepdims=True), train_co_flat)
    test_co_clean = np.where(np.isnan(test_co_flat), np.nanmean(test_co_flat, axis=1, keepdims=True), test_co_flat)
    
    # Decompose
    n_components = 5
    print(f"Fitting PCA (EOF) with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    
    # Fit on train
    train_coefs = pca.fit_transform(train_co_clean) # (T_train, n_components)
    
    print("Fitting SARIMA models on temporal modes...")
    # S5P data is 5-day composites. 365 / 5 = 73 steps per year.
    # Seasonal order m=73
    
    # Create regular time index for SARIMA
    ts_train = pd.Series(train_dates)
    
    # We will fit a simple AR(1) or SARIMA(1,0,0)(1,0,0,73) for each component
    models = []
    for i in range(n_components):
        series = pd.Series(train_coefs[:, i], index=train_dates)
        # Resample to regular 5-day to fill any missing gaps
        series_reg = series.resample('5D').mean().interpolate()
        
        # Simple ARIMA for speed, with annual seasonality (m=73)
        # Using a lightweight model for this demonstration
        model = sm.tsa.statespace.SARIMAX(
            series_reg, 
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 73),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        models.append((res, series_reg.index[-1]))
        
    print("Forecasting Test Set (2024-2025)...")
    # We want to forecast specifically for the test_dates.
    # We can forecast out to the max test date.
    max_test_date = max(test_dates)
    
    pred_coefs = np.zeros((T_test, n_components))
    
    for i, (res, last_train_dt) in enumerate(models):
        # Forecast from last_train_dt to max_test_date
        steps = (max_test_date - last_train_dt).days // 5 + 1
        forecast = res.forecast(steps=steps)
        # Match test_dates
        for j, dt in enumerate(test_dates):
            # Find closest date in forecast index
            idx = np.argmin(np.abs(forecast.index - dt))
            pred_coefs[j, i] = forecast.iloc[idx]
            
    print("Reconstructing CO fields...")
    pred_co_clean = pca.inverse_transform(pred_coefs) # (T_test, H*W)
    
    # Calculate R2 only on valid test pixels
    valid_mask = ~np.isnan(test_co_flat)
    
    r2 = r2_score(test_co_flat[valid_mask], pred_co_clean[valid_mask])
    rmse = np.sqrt(mean_squared_error(test_co_flat[valid_mask], pred_co_clean[valid_mask]))
    
    print(f"=== CO Long-Range EOF+SARIMA Performance ===")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.4e}")
    
    # Compare with Climatology
    clim_mean_all = np.load("classical_ml/data/climatology_mean.npy") # (12, 3, H, W)
    clim_mean_co = clim_mean_all[:, 1, :, :] # (12, H, W)
    
    clim_preds = np.zeros_like(test_co_flat)
    for j, dt in enumerate(test_dates):
        month_idx = dt.month - 1
        clim_preds[j] = clim_mean_co[month_idx].flatten()
        
    r2_clim = r2_score(test_co_flat[valid_mask], clim_preds[valid_mask])
    rmse_clim = np.sqrt(mean_squared_error(test_co_flat[valid_mask], clim_preds[valid_mask]))
    
    print(f"Climatology R²   : {r2_clim:.4f}")
    print(f"Climatology RMSE : {rmse_clim:.4e}")

if __name__ == "__main__":
    main()
