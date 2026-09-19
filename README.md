# AirQ-Astra: Physically Constrained Multi-Pollutant Forecasting & Downscaling

This workspace is a clean-room implementation plan for an India-focused satellite air-quality project using Sentinel-2, Sentinel-5P, ERA5 meteorology, optional CPCB stations, and optional facility metadata.

## Primary task
Forecast the next Sentinel-5P observation for NO2, CO, and SO2, while producing a physically constrained finer spatial allocation guided by Sentinel-2 and meteorology.

## Core scientific rule
There is no true 100 m gas ground truth. Fine maps are model-derived spatial allocations. All quantitative supervision and headline metrics must be computed against native/coarse Sentinel-5P observations or independent ground stations where available.

## Start here
1. Read `AGENTS.md`.
2. Read `docs/00_PROJECT_OVERVIEW.md` through `docs/14_ASTRA_EXECUTION_RULES.md` in order.
3. Paste data exactly as described in `docs/03_DATA_PASTE_GUIDE.md`.
4. Run the data audit before writing or training any model.
5. Execute experiments in `docs/12_EXPERIMENT_LADDER.md` sequentially.

## Non-negotiable gates
- No temporal leakage.
- No fake 5-day sequences across long gaps.
- No bilinear-interpolated Sentinel-5P used as fine ground truth.
- No loss on nodata pixels.
- No claim that Sentinel-2 directly measures NO2/CO/SO2.
- No claim of high-resolution accuracy without independent high-resolution truth.
- Every architecture change must be benchmarked against a frozen baseline under the same split and seed protocol.
