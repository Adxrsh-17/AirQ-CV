# AirQ-Astra Dataset V2 generation design

**Phase:** 0C (design only; no export and no model training)
**Configuration:** [`configs/dataset_v2.yaml`](../configs/dataset_v2.yaml)
**Generator:** [`scripts/dataset_generation/`](../scripts/dataset_generation/) and `src/airq/data/generation/`

## Gate decision

**PHASE 0C DESIGN: BLOCKED — AWAITING ROI ADOPTION.**

The recipe below is explicit, deterministic and testable. The old `data/raw/`
archive is immutable and its intended ROI was not recovered in Phase 0B. The
candidate rectangle is recorded in the configuration, but it is not silently
promoted to the verified study ROI. A reviewer must provide the original ROI
polygon or explicitly adopt the candidate rectangle before an Earth Engine
task can be submitted. `design_gate.py`, `export_s2.py` and `export_s5p.py`
therefore remain plan-only commands and return a non-zero status.

The SSL4EO checkpoint choice is deliberately outside this data-generation gate;
the Phase 1 encoder must separately verify its input contract.

## Source and processing contract

### Sentinel-2

Use the official Earth Engine `COPERNICUS/S2_SR_HARMONIZED` collection
(Level-2A surface reflectance). The catalogue documents the 2022 processing
baseline harmonization, 0.0001 reflectance scale, and native sampling:

| Stored physical band order | Native sampling | Units after scaling |
| --- | ---: | --- |
| B2, B3, B4, B8 | 10 m | dimensionless reflectance |
| B5, B6, B7, B11, B12 | 20 m | dimensionless reflectance |

The generator scales digital numbers by `0.0001` and retains the nine bands in
that exact order. Native 10 m inputs are area-weighted averaged onto a 20 m
working grid; QA masks use nearest-neighbour resampling. No clipping or finite
filling is performed.

For each image, join `COPERNICUS/S2_CLOUD_PROBABILITY` by `system:index`.
Reject an image with no join, with `CLOUDY_PIXEL_PERCENTAGE > 80`, or with no
valid edge mask from B8A/B9. A pixel is accepted only when all nine physical
bands are finite and valid, cloud probability is `< 40` (strict), and SCL is
one of 4 (vegetation), 5 (bare soil), or 6 (water). The scene threshold is
inclusive (`<= 80`). These rules are project policy and are recorded in every
product sidecar.

Within a five-day interval, reduce accepted datatakes using a pixelwise median
on the 20 m grid. Compute indices **after** this temporal reduction using the
custom expression below, so finite negative reflectances are not implicitly
masked by an implementation-specific normalized-difference helper:

* `NDVI = (B8 - B4) / (B8 + B4)`
* `NDBI = (B11 - B8) / (B11 + B8)`
* `NDMI = (B8 - B11) / (B8 + B11)`

An index is invalid only when either input is non-finite or its denominator is
exactly zero. Index contributor count is the minimum of the two input-band
support counts. The output has 12 bands: the nine physical bands followed by
NDVI, NDBI and NDMI.

### Sentinel-5P

Use the official offline Level-3 Earth Engine collections. Level-3 assets are
orbit-gridded products derived from Level-2 data; their upstream validity mask
is retained, and the source asset ID and processing properties are copied to
the sidecar. No unavailable L2 QA raster is reconstructed locally.

| Pollutant | Collection and band | Semantics and units | Source validity rule |
| --- | --- | --- | --- |
| NO2 | `COPERNICUS/S5P/OFFL/L3_NO2`, `tropospheric_NO2_column_number_density` | tropospheric vertical column, `mol/m^2` | Earth Engine ingestion uses `tropospheric_NO2_column_number_density_validity > 75`; local mask is source mask + finite value |
| CO | `COPERNICUS/S5P/OFFL/L3_CO`, `CO_column_number_density` | vertically integrated/total column, `mol/m^2` | upstream HARP validity `> 50`; local mask is source mask + finite value |
| SO2 | `COPERNICUS/S5P/OFFL/L3_SO2`, `SO2_column_number_density` | default ground-level vertical column, `mol/m^2`; not the separate 15 km band | source-adjusted QA: snow/ice `<0.5`, polluted AMF `>0.1`, total column `>-0.001 mol/m^2`, QA `>0.5`, cloud fraction `<0.3`, solar zenith `<60°`, then HARP validity `>50` |

The L3 catalogues expose no pollutant-specific QA band suitable for a second
local threshold. `QA band/field` in a sidecar is therefore the named upstream
validity rule above, plus the exported source mask. Cloud and retrieval rules
are those source rules; no generic cloud filter is invented for CO or NO2.

For every pollutant, finite source-valid exact zeros are retained as measured
values. Finite negative values passed by the source rule are retained; there is
no `max(0)`, clipping, `unmask(0)`, `fillna(0)`, or `nan_to_num(..., 0)`.
SO2 values below the source's `-0.001 mol/m^2` screening threshold are absent
because the official ingestion rule rejects them, not because the generator
clips them. Missing values are stored as IEEE NaN with a separate uint8 mask.

Aggregate each accepted orbit onto the 5,000 m EPSG:32644 support grid using a
masked area-weighted mean. A cell needs at least 50% covered source support.
Reduce accepted orbit values in `[start_time, end_time)` using a pixelwise
arithmetic mean. `count` is the number of distinct accepted orbit/source image
groups contributing to that output cell (uint16), not the number of L2 pixels.

## Temporal and spatial contract

Windows start at `2019-01-01T00:00:00Z`, have five-day half-open intervals
`[start_time, end_time)`, and use `system:time_start` in UTC for source
assignment. The label is the composite covering that interval; it is not an
instantaneous five-day-ahead observation. The requested end is
`2025-01-01T00:00:00Z`; 438 complete windows end at 2024-12-30 and the final
two-day tail is omitted and reported. Processing/export timestamps are metadata
only and never used for temporal assignment.

The candidate study ROI is the audited S2 extent
`[79.94916197135329, 12.799196168134944, 80.45042189989198, 13.300456096673637]`
in EPSG:4326. It is explicitly labelled a candidate because Phase 0B did not
recover the original polygon. The output rectangle is the densified UTM
EPSG:32644 envelope snapped outward to 5 km multiples:
`[385000, 1410000, 445000, 1475000]` metres.

| Grid | Resolution | Dimensions | Role |
| --- | ---: | ---: | --- |
| S2 | 20 m | 3,000 x 3,250 | reflectance/indices working and stored grid |
| S5P | 5,000 m | 12 x 13 | native-support supervision grid |
| fine allocation | 500 m | 120 x 130 | model-derived allocation only; never ground truth |

The S5P grid is a regular export support grid, not a set of original TROPOMI
footprint polygons. A 500 m allocation grid is recommended over 100 m because
TROPOMI nominal support is several kilometres; 500 m limits the number of
unsupported degrees of freedom while still allowing urban-scale allocation.
It does not create 500 m gas truth. Any later fine-grid output must be evaluated
after aggregation back to the 5 km support grid.

## Product, masks and manifest

Each generated product directory contains `values.tif`, `valid.tif`,
`count.tif`, and a JSON sidecar. Values are float32 with NaN nodata; masks are
uint8 (0/1) with no nodata; counts are uint16 (zero means no contributors).
Products are append-only under `data/raw_v2/`; an existing path is an error.
The machine-readable sidecar and manifest field contract is versioned at
`provenance/dataset_v2/metadata_schema.yaml`.

The V2 manifest (`airq.manifest-v2/0.1`) has one row per scheduled window and
includes `window_id`, UTC start/end, explicit S2/S5P paths and sidecars, status,
per-band validity fractions, denominator definition, source collection/version
references, config SHA-256, generator commit, creation time and recipe
reference. A row is `PLANNED` until sidecars are verified; synthetic fixtures
may be marked `SYNTHETIC_PARTIAL` or `SYNTHETIC_COMPLETE`. Production exports
must never be inferred from filenames alone.

## Generation and verification stages

The modules intentionally separate collection preparation, QA masking,
per-observation spatial aggregation, temporal reduction, task submission,
download/synchronization, export verification and manifest append. The current
submission and synchronization functions raise an error by design. A future
implementation may enable them only after ROI adoption and a reviewed pilot.
No credentials or tokens are embedded; an authenticated Earth Engine project,
Cloud Storage/Drive destination, and local raster tooling will be required.

## Workload estimate

The deterministic plan contains 438 windows, 876 logical products, 2,628 TIFFs,
and 3,504 candidate remote value/auxiliary tasks. Uncompressed value + mask +
count storage is approximately 358.722 GB (334.09 GiB) for S2 and 1.435 MB for
S5P. Compression is intentionally unmeasured until a small approved pilot;
the full archive must not be exported from this design gate.

## Official references

* [Sentinel-2 SR Harmonized catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
* [Sentinel-2 Cloud Probability catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY)
* [Sentinel-5P OFFL NO2 catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2)
* [Sentinel-5P OFFL CO catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_CO)
* [Sentinel-5P OFFL SO2 catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_SO2)
* [Earth Engine half-open `filterDate` semantics](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-filterdate)

These references establish current catalogue semantics, not the provenance of
the old archive. The V2 archive will carry the exact asset IDs and properties
used for each export.
