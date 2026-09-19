# Deep Learning 150-day Direct Multi-Horizon Forecasting

## Model: ConvLSTM-based temporal-spatial forecasting baseline
- History: H=12 (12 x 5-day windows = 60 days)
- Forecast: K=30 (30 x 5-day windows = 150 days)
- Training: channel-wise train normalization, GPU training, direct multi-horizon forecast
