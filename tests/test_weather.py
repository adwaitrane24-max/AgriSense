"""Tests for the Open-Meteo weather client (fake fetcher, no network)."""

from datetime import date
from typing import Any
from urllib.error import URLError

import pytest

from leafcheck import weather


@pytest.fixture(autouse=True)
def _empty_cache() -> None:
    weather.clear_cache()


def canned_open_meteo() -> dict:
    """A trimmed copy of a real Open-Meteo daily response."""
    dates = [f"2026-10-{d:02d}" for d in range(30, 32)] + [
        f"2026-11-{d:02d}" for d in range(1, 6)
    ]
    return {
        "latitude": 18.5,
        "longitude": 73.875,
        "timezone": "Asia/Kolkata",
        "daily": {
            "time": dates,
            "temperature_2m_max": [31.24, 30.9, 30.1, 29.8, 30.5, 31.0, 31.2],
            "temperature_2m_min": [19.06, 19.4, 20.0, 19.8, 19.1, 18.7, 18.9],
            "relative_humidity_2m_max": [88, 90, 92, 85, 80, 78, 77],
            "relative_humidity_2m_mean": [64, 66, 70, 61, 58, 55, 54],
            "precipitation_sum": [0.0, 1.25, 4.4, 0.0, 0.0, 0.0, None],
            "precipitation_probability_max": [10, 35, 60, 15, 5, 3, 0],
            "wind_speed_10m_max": [11.3, 12.0, 14.61, 10.1, 9.8, 9.5, 9.0],
        },
    }


class CountingFetch:
    """Fake fetcher that records URLs and returns a fixed payload."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.urls.append(url)
        return self.payload


def test_parses_seven_days() -> None:
    fetch = CountingFetch(canned_open_meteo())
    result = weather.get_forecast(18.52, 73.86, fetch=fetch)
    assert result["source"] == "live"
    days = result["days"]
    assert len(days) == 7
    assert days[0] == {
        "date": "2026-10-30",
        "t_max": 31.2,
        "t_min": 19.1,
        "humidity_max": 88.0,
        "humidity_mean": 64.0,
        "rain_mm": 0.0,
        "rain_prob": 10.0,
        "wind_max": 11.3,
    }
    assert days[2]["rain_mm"] == 4.4
    assert days[2]["wind_max"] == 14.6
    assert days[6]["rain_mm"] is None  # null in the response


def test_url_contains_required_parameters() -> None:
    url = weather.build_url(18.52, 73.86)
    assert url.startswith("https://api.open-meteo.com/v1/forecast?")
    assert "latitude=18.5200" in url and "longitude=73.8600" in url
    assert "timezone=Asia%2FKolkata" in url and "forecast_days=7" in url
    for field in weather.DAILY_FIELDS:
        assert field in url


def test_network_error_returns_offline_seven_days() -> None:
    def failing(url: str) -> Any:
        raise URLError("no internet")

    result = weather.get_forecast(18.52, 73.86, fetch=failing)
    assert result["source"] == "offline"
    assert len(result["days"]) == 7
    assert result["days"][0]["date"] == date.today().isoformat()
    assert all(day["t_max"] is not None for day in result["days"])


def test_bad_payload_returns_offline() -> None:
    result = weather.get_forecast(18.52, 73.86, fetch=CountingFetch({"error": True}))
    assert result["source"] == "offline"


def test_second_call_hits_cache() -> None:
    fetch = CountingFetch(canned_open_meteo())
    first = weather.get_forecast(18.52, 73.86, fetch=fetch)
    second = weather.get_forecast(18.521, 73.859, fetch=fetch)  # rounds to same key
    assert len(fetch.urls) == 1
    assert second == first


def test_offline_result_is_not_cached() -> None:
    def failing(url: str) -> Any:
        raise TimeoutError

    weather.get_forecast(18.52, 73.86, fetch=failing)
    fetch = CountingFetch(canned_open_meteo())
    assert weather.get_forecast(18.52, 73.86, fetch=fetch)["source"] == "live"
    assert len(fetch.urls) == 1


def test_offline_dates_follow_given_day() -> None:
    result = weather.load_offline_forecast(today=date(2026, 1, 30))
    assert [d["date"] for d in result["days"]][:3] == [
        "2026-01-30",
        "2026-01-31",
        "2026-02-01",
    ]
