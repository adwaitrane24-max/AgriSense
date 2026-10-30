"""Tests for the SoilGrids client and USDA texture classifier (no network)."""

from typing import Any, Optional
from urllib.error import URLError

import pytest

from leafcheck import soil
from leafcheck.districts import SOIL_TYPES


@pytest.fixture(autouse=True)
def _empty_cache() -> None:
    soil.clear_cache()


def _layer(name: str, mean: Optional[int], d_factor: int = 10) -> dict:
    return {
        "name": name,
        "unit_measure": {"d_factor": d_factor, "mapped_units": "x", "target_units": "y"},
        "depths": [
            {"range": {"top_depth": 0, "bottom_depth": 5}, "label": "0-5cm", "values": {"mean": mean}}
        ],
    }


def canned_soilgrids(**overrides: Optional[int]) -> dict:
    """Shape of a real SoilGrids /properties/query reply for a black-soil point."""
    means = {"phh2o": 76, "clay": 480, "sand": 220, "silt": 300, "soc": 135}
    means.update(overrides)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [73.86, 18.52]},
        "properties": {"layers": [_layer(k, v) for k, v in means.items()]},
    }


class CountingFetch:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls = 0

    def __call__(self, url: str) -> Any:
        self.calls += 1
        return self.payload


def test_live_values_are_converted() -> None:
    result = soil.get_soil(18.52, 73.86, fetch=CountingFetch(canned_soilgrids()))
    assert result["source"] == "live"
    assert result["ph"] == pytest.approx(7.6)
    assert result["clay_pct"] == pytest.approx(48.0)
    assert result["organic_carbon_gkg"] == pytest.approx(13.5)
    total = result["clay_pct"] + result["sand_pct"] + result["silt_pct"]
    assert total == pytest.approx(100, abs=1)
    assert result["texture"] == "clay"
    assert result["soil"] in SOIL_TYPES and result["district"] == "pune"


def test_url_has_all_properties() -> None:
    url = soil.build_url(18.52, 73.86)
    for prop in soil.PROPERTIES:
        assert f"property={prop}" in url
    assert "depth=0-5cm" in url and "value=mean" in url
    assert "lat=18.5200" in url and "lon=73.8600" in url


@pytest.mark.parametrize(
    ("sand", "silt", "clay", "expected"),
    [
        (92, 4, 4, "sand"),
        (82, 12, 6, "loamy_sand"),
        (65, 25, 10, "sandy_loam"),
        (40, 40, 20, "loam"),
        (20, 65, 15, "silt_loam"),
        (5, 88, 7, "silt"),
        (60, 13, 27, "sandy_clay_loam"),
        (33, 34, 33, "clay_loam"),
        (10, 55, 35, "silty_clay_loam"),
        (50, 5, 45, "sandy_clay"),
        (5, 50, 45, "silty_clay"),
        (20, 20, 60, "clay"),
    ],
)
def test_usda_texture_classes(sand: float, silt: float, clay: float, expected: str) -> None:
    assert soil.usda_texture(sand, silt, clay) == expected


def test_texture_rescales_inputs() -> None:
    assert soil.usda_texture(4, 4, 2) == soil.usda_texture(40, 40, 20) == "loam"


def test_every_texture_has_three_language_names() -> None:
    assert len(soil.TEXTURE_NAMES) == 12
    for names in soil.TEXTURE_NAMES.values():
        assert set(names) == {"en", "hi", "mr"}


def test_failing_fetch_falls_back_to_district() -> None:
    def failing(url: str) -> Any:
        raise URLError("down")

    result = soil.get_soil(18.52, 73.86, fetch=failing)
    assert result["source"] == "district"
    assert result["soil"] == "black_cotton"
    assert result["ph"] is None and result["texture"] is None


def test_null_values_fall_back_to_district() -> None:
    payload = canned_soilgrids(phh2o=None, clay=None)
    result = soil.get_soil(16.99, 73.30, fetch=CountingFetch(payload))
    assert result["source"] == "district"
    assert result["soil"] == "laterite"  # nearest is Ratnagiri


def test_live_result_is_cached() -> None:
    fetch = CountingFetch(canned_soilgrids())
    soil.get_soil(18.52, 73.86, fetch=fetch)
    soil.get_soil(18.52, 73.86, fetch=fetch)
    assert fetch.calls == 1
