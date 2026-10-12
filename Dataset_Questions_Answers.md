# Dataset Analysis Questions & Answers

**4) What are the fundamental characteristics and statistical properties of the dataset that need to be understood before modelling?**

Before modeling, the following core characteristics and statistical properties of the Sentinel-5P dataset (2019-2024) must be addressed:
* **Radical Distribution Divergence & Dynamic Range:** The three gases cannot be modeled using a single uniform loss function.
  * **CO (Carbon Monoxide):** Is well-behaved, diffuse, and near-Gaussian (Skewness = -0.18). Linear L1/L2 regression works perfectly.
  * **NO₂ (Nitrogen Dioxide) & SO₂ (Sulfur Dioxide):** Are severely skewed (Skewness > 3.2, Kurtosis > 26). The data is heavily right-skewed with extreme industrial plume spikes in the tail (SO₂ maximum is 34.1x the mean). 
* **Extreme Sparsity & Missing Data:** SO₂ suffers from 51.02% invalid data (cloud cover/retrieval failures requiring `qa_value > 0.5` masking). Furthermore, most valid SO₂ pixels are near-zero, making it highly sparse.
* **Loss Function Implications:** Due to the extreme leptokurtic distribution of NO₂ and SO₂, standard Mean Squared Error (MSE) will fail. It forces the gradient descent to focus only on single outlier pixels while neglecting 95% of the background. Logarithmic compression or treating SO₂ as an extreme-event hazard classification (Hurdle Model) is strictly required.

---

**7) What are the trends observed for each gas over the available time period?**

Over the 6-year period (2019-2024), the following long-term and spatial trends are observed:
* **Spatial Inequality & Hotspots:**
  * **NO₂ (Gini = 0.1598):** Highly concentrated along transport arteries, urban hubs, and industrial corridors (e.g., Coimbatore-Tirupur corridor, Chennai).
  * **SO₂ (Gini = 0.1139):** Exhibits intense point-source clusters tightly localized around mega-watt coal-fired thermal power stations and refineries (e.g., North Chennai Coastal Industrial Corridor, Neyveli Lignite Mining, Thoothukudi Port).
  * **CO (Gini = 0.0263):** Extremely homogeneous. CO acts as a dispersed regional background gas rather than forming sharp localized plumes.
* **Temporal Autocorrelation (Persistence):**
  * **NO₂** maintains strong spatial persistence (r > 0.46) even up to 60 days. This isn't because the gas survives for 60 days, but because the underlying emission sources (highways, factories) are stationary and continuous.
  * **SO₂** has a very rapid memory decay (e-folding time of ~5 days), confirming its episodic and intermittent industrial release nature.

---

**8) What seasonal or monthly trends can be identified from the dataset?**

The dataset exhibits distinct meteorological and seasonal cycles affecting gas dispersion and accumulation:
* **Winter (Jan–Feb):** Exhibits the highest concentrations for **CO** (3.782 × 10⁻² mol/m²) and **SO₂** (1.500 × 10⁻⁴ mol/m²). This is caused by shallow boundary layers, nocturnal thermal inversions, and low wind ventilation, which trap pollutants close to the surface.
* **Pre-Monsoon / Summer (Mar–May):** **NO₂** reaches its absolute peak (April). This is driven by intense pre-monsoon solar irradiance that accelerates photochemical NOx cycling, combined with peak power generation demand.
* **Southwest Monsoon (Jun–Sep):** All gases experience significant depletion due to strong wet scavenging (rain washout) and high boundary layers. Most notably, **CO** sees a dramatic 33% drop due to clean marine air advection blowing in from the Indian Ocean.
