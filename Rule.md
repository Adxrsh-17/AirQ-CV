TODO — Land-Use-Informed Pollution Baseline

0. Do not drop any meaningful data in the name of hardware constraints, local laptop performance, or aggressive preprocessing/feature extraction. This is not a time series model.

- Critical Preprocessing Audit :
  The raw GeoTIFFs exported from Google Earth Engine are already high-resolution (558 x 558 pixels, in order of ~10^2m, 12 spectral bands).
  Aggressive downsampling occurred inside `align_tn.py` via `R = 8` (8 S2 sub-pixels per S5P cell). Because Sentinel-5P has a coarse resolution (~5.5 km, 13 x 13 pixels over Chennai), the script downscaled the 558 x 558 Sentinel-2 optical images down to 13 x 8 = 104 x 104 pixels (~700m per cell) 
  
  At lower resolution, individual road networks, vehicular traffic corridors, and narrow industrial buildup become severely blurred into broad background averages, destroying the fine-grained land-use predictive signal. This downsampling violates Rule 0 and is strictly prohibited. Instead explore methods that utilise full res data (Prefer high res data so high quality learning) while ensuring efficiency and objectivity. (THIS NEEDS TO BE ADDRESSED BEFORE CONTINUING OTHERE SUGESSIONS WERE IMPLEMENTED)

1. Objective & Architectural Context

Build a pure spatial baseline model that learns pollution signatures inferred from land use using Sentinel-2 surface reflectance, rather than memorizing temporal or climatic patterns.

- Why this exists (Legacy Failure Mode): Previous iterations overfitted on temporal autocorrelation and meteorology, acting effectively as pure time-series models while ignoring surface/land-use significance. Furthermore, over-downsampling S2 optical features to 700m blinded the spatial encoder to fine-grained land use.
- How this fits the Core Model: We already have a planned ML/time-series counterpart to handle temporal dynamics in the final core architecture. This stage isolates the spatial encoder. It must prove it can extract static/quasi-static land-use pollution signatures on its own before we fuse it with the temporal counterpart.


---

2. Non-Negotiable Rules

1. Land-use signal is the sole objective of this stage. The model must map spatial/surface characteristics (industrial zones, road networks, urban density, vegetation) to baseline pollution levels.
2. Zero temporal/climate shortcuts in the baseline. Do not feed lag features, timestamps, or meteorological variables into the Sentinel-2 baseline branch.
3. Do not equate high time-series $R^2$ with success. A lower, honest spatial $R^2$ on held-out locations is acceptable and expected; a high $R^2$ driven by temporal leakage will be rejected.
4. Use temporally aggregated targets to stabilize the spatial signal. Frame the baseline target over monthly, quarterly, or long-term means (following the national-no2-cnn and Khan et al. aggregation approaches) so high-frequency weather noise does not drown out the land-use signature.
5. Document Sentinel-5P constraints. Where Sentinel-5P column density is used as supervision, a target, or a secondary fusion branch, explicitly document its spatial resolution gap and column-to-ground mismatch.
6. Follow rule.md strictly. Any deviation from rule.md must be logged and justified before running experiments.
7. Strictly Preserve Fine-Grained Optical Resolution: Sentinel-2 optical features must NOT be aggressively downscaled to coarse resolutions (such as the legacy R=8 ~700m downsampling in align_tn.py). The spatial encoder requires native or fine-scale resolution (e.g., 60m–100m, retaining the 558 x 558 grid) so that emission hot spots (power plants, refineries, major highway corridors) retain distinct structural contrast rather than being blurred into ambient background noise.


---

3. Action Items (TODO)

- [ ] Review Context & Rules: Read rule.md, the legacy failure notes, and the reference papers/repos below before writing or modifying model code.
- [ ] Lock Problem Definition: Define the target pollutant ($\text{NO}_2$), spatial patch size (e.g., $120 \times 120\text{ px}$ at $10\text{ m}$ resolution / $1.2 \times 1.2\text{ km}$), temporal aggregation window (daily vs. monthly/quarterly mean), and Sentinel-2 spectral bands.
- [ ] Enforce Leak-Free Spatial Splits: Implement spatial block cross-validation or distance-buffered splits so training and validation/test Sentinel-2 patches never overlap geographically.
- [ ] Build Pure Sentinel-2 Baseline: Train a clean, reproducible CNN/spatial baseline using only Sentinel-2 surface reflectance to predict the aggregated pollutant target.
- [ ] Run Shortcut & Fusion Ablations:
  - Config A: Sentinel-2 only (Pure Land-Use Baseline — Primary Focus)
  - Config B: Temporal / Climate / S5P only (Shortcut / Atmospheric Baseline)
  - Config C: Combined inputs (Evaluate genuine marginal gain from Sentinel-2)
- [ ] Verify Land-Use Learning: Provide direct evidence that the model learned surface patterns (e.g., performance across unseen spatial blocks, error breakdown by land-cover class, or spatial attribution/Grad-CAM checks).
- [ ] Document Failure Boundaries (Old vs. New):
  - Document where the older time-series approach failed (spatial blindness / unseen locations).
  - Document where this new land-use change still does not help (e.g., transient meteorological trapping, wind transport, short-term traffic spikes) so we know exactly what the core ML counterpart must cover next.


---

4. Acceptance Criteria

The baseline is accepted only if:

1. Experiments demonstrate that Sentinel-2 surface reflectance provides predictive signal on spatially held-out locations with zero temporal/meteorological inputs.
2. When combined with atmospheric/temporal inputs in ablations, the Sentinel-2 branch demonstrably improves spatial generalization rather than being ignored by the network.
3. Failure cases where land-use alone is insufficient are clearly logged before integrating the ML/time-series counterpart.


---

5. References

- Project Rulebook (rule.md): Local source of truth for engineering and modeling constraints.
- Reference Paper (Land-Use CNN): Frontiers in Environmental Science (2023) — https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2023.1285471/full
- Reference Codebase (national-no2-cnn): https://github.com/eltonc01/national-no2-cnn
- Multisource Fusion & Temporal Aggregation Reference: Solaiman Khan et al. (2026), High-resolution ground $\text{NO}_2$ estimation at hyperlocal level using deep learning with Sentinel-2 and Sentinel-5P data (refer to project summary for daily vs. monthly/quarterly $R^2$ scaling and patch setup; verify full bibliographic details before formal citation).


---

6. Scope Boundary

This document defines the overall objective, stage boundaries, and validation guardrails. Detailed preprocessing pipelines, hyperparameter tables, and core-model fusion architecture belong in separate implementation docs.
-Sentinel-2-only inputs do not automatically prove that the model learned land-use pollution signatures. It could learn other spatial correlations. Your ablations, held-out spatial validation, and land-cover analysis help test that distinction.
-Monthly or quarterly target aggregation stabilizes the spatial signal but does not, by itself, remove meteorological or other influences.