"""Synthetic V2 product/manifest I/O tests; never touch data/raw."""

from pathlib import Path

import numpy as np
import pytest

from airq.data.generation.contracts import design_plan, load_config
from airq.data.generation.manifest import build_manifest, write_manifest
from airq.data.generation.products import verify_product, write_product, write_recipe

pytestmark = pytest.mark.smoke
ROOT = Path(__file__).resolve().parents[1]


def _grid(config):
    return design_plan(config)["grids"]["s5p"]


def _source(config, sensor, window):
    if sensor == "s2":
        collections = [config["s2"]["collection"]]
    else:
        collections = [config["s5p"]["pollutants"][gas]["collection"] for gas in ("NO2", "CO", "SO2")]
    return [{
        "source_id": f"synthetic-{sensor}-{index}",
        "collection": collection,
        "acquired_at": window["start_time"],
        "observation_group": f"orbit-{index}",
        "version_properties": {"PROCESSOR_VERSION": "synthetic"},
    } for index, collection in enumerate(collections, 1)]


def test_product_sidecar_binds_grid_units_masks_zero_and_negative(tmp_path):
    config = load_config(ROOT / "configs/dataset_v2.yaml")
    recipe = write_recipe(tmp_path, config, {"source_tree_sha256": "a" * 64, "dependencies": {"numpy": "test"}})
    window = {"window_id": "20190101", "start_time": "2019-01-01T00:00:00Z", "end_time": "2019-01-06T00:00:00Z"}
    grid = _grid(config)
    values = np.full((3, grid["height"], grid["width"]), np.nan, dtype=np.float32)
    values[:, 0, 0] = [0.0, -0.001, 2.0]
    valid = np.zeros_like(values, dtype=np.uint8)
    valid[:, 0, 0] = 1
    counts = valid.astype(np.uint16)
    relative = write_product(tmp_path, recipe, "s5p", window, grid, values, valid, counts,
                             _source(config, "s5p", window))
    metadata = verify_product(tmp_path, relative, expected_grid=grid)
    assert metadata["units"] == ["mol/m^2"] * 3
    assert metadata["valid_fraction_export_rectangle"]["NO2"] == pytest.approx(1 / (grid["width"] * grid["height"]))


def test_manifest_is_planned_then_synthetic_complete(tmp_path):
    config = load_config(ROOT / "configs/dataset_v2.yaml")
    recipe = write_recipe(tmp_path, config, {"source_tree_sha256": "b" * 64, "dependencies": {"numpy": "test"}})
    window = {"window_id": "20190101", "start_time": "2019-01-01T00:00:00Z", "end_time": "2019-01-06T00:00:00Z"}
    grid = _grid(config)
    values = np.zeros((3, grid["height"], grid["width"]), dtype=np.float32)
    valid = np.ones_like(values, dtype=np.uint8)
    counts = np.ones_like(values, dtype=np.uint16)
    sidecar = write_product(tmp_path, recipe, "s5p", window, grid, values, valid, counts,
                            _source(config, "s5p", window))
    rows = build_manifest(config, tmp_path, [sidecar])
    row = next(r for r in rows if r["window_id"] == "20190101")
    assert row["record_state"] == "SYNTHETIC_PARTIAL"
    assert row["s5p_status"] == "ok"
    output = write_manifest(tmp_path, "data/raw_v2/manifest.csv", rows)
    assert output.exists() and output.read_text().startswith("schema_version,")
