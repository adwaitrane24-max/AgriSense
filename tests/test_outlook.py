"""Tests for the Field Outlook page and API (weather and soil are faked, no network)."""

from typing import Any

import pytest
from flask.testing import FlaskClient

from leafcheck import outlook, soil, weather
from leafcheck.i18n import UI_STRINGS

PUNE = (18.52, 73.86)


def cool_wet_week() -> dict[str, Any]:
    days = [
        {"date": f"2026-11-0{i + 1}", "t_max": 22.0, "t_min": 14.0, "humidity_max": 96.0,
         "humidity_mean": 88.0, "rain_mm": 25.0 if i == 1 else 6.0, "rain_prob": 80, "wind_max": 12.0}
        for i in range(7)
    ]
    return {"source": "live", "days": days}


@pytest.fixture(autouse=True)
def fake_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather, "get_forecast", lambda lat, lon, **_: cool_wet_week())
    monkeypatch.setattr(soil, "get_soil", lambda lat, lon, **_: {
        "source": "live", "ph": 7.6, "clay_pct": 48.0, "sand_pct": 22.0, "silt_pct": 30.0,
        "organic_carbon_gkg": 9.5, "texture": "clay"})


def test_build_outlook_names_district_and_ranks_risks() -> None:
    data = outlook.build_outlook(*PUNE, "en")
    assert data["district"]["slug"] == "pune"
    assert len(data["weather"]["days"]) == 7
    levels = [r["level"] for r in data["risks"]]
    assert levels == sorted(levels, key=["high", "medium", "low"].index)
    blight = next(r for r in data["risks"] if r["label"] == "Potato___Late_blight")
    assert blight["level"] == "high" and blight["peak_text"]
    assert data["tips"][-1]["key"] == "contact_kvk"
    assert data["soil"]["texture_name"]


def test_outlook_is_translated() -> None:
    data = outlook.build_outlook(*PUNE, "mr")
    assert data["lang"] == "mr"
    assert any("ऀ" <= ch <= "ॿ" for ch in data["district"]["name"])
    assert any("ऀ" <= ch <= "ॿ" for ch in data["risks"][0]["reason_text"])


def test_soil_falls_back_to_district_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(soil, "get_soil", lambda lat, lon, **_: {"source": "district", "soil": None})
    data = outlook.build_outlook(*PUNE, "en")
    assert data["soil"]["source"] == "district"
    assert data["soil"]["type_name"]


def test_district_options_cover_all_districts() -> None:
    options = outlook.district_options("hi")
    assert len(options) == 36
    assert {"slug", "name", "lat", "lon"} <= set(options[0])


def test_outlook_page_renders_with_district_list(client: FlaskClient) -> None:
    page = client.get("/outlook?lang=hi").get_data(as_text=True)
    assert UI_STRINGS["hi"]["map_title"] in page
    assert 'id="district-select"' in page
    assert page.count("<option value=") == 37  # placeholder + 36 districts


def test_outlook_api_returns_json(client: FlaskClient) -> None:
    response = client.get("/api/outlook?lat=18.52&lon=73.86&lang=en")
    assert response.status_code == 200
    body = response.get_json()
    assert body["district"]["slug"] == "pune"
    assert body["risks"] and body["tips"]


@pytest.mark.parametrize("query", ["lat=28.6&lon=77.2", "lat=abc&lon=73", "lon=73.8"])
def test_outlook_api_rejects_points_outside_maharashtra(client: FlaskClient, query: str) -> None:
    response = client.get(f"/api/outlook?{query}&lang=en")
    assert response.status_code == 400
    assert response.get_json()["error"] == UI_STRINGS["en"]["map_outside"]
