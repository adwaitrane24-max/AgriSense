"""7-day weather forecast from Open-Meteo (free, no API key), with fallbacks.

Flow of `get_forecast(lat, lon)`:
    1. Look in a TTL cache keyed by (lat, lon) rounded to 2 decimals (~1 km).
    2. Miss -> call Open-Meteo, parse the "daily" arrays into 7 day dicts,
       store in the cache for 30 minutes.
    3. Any network or parse error -> return a bundled offline sample (Pune,
       monsoon-like) marked `"source": "offline"`, so the page still works
       without internet (e.g. during a demo). Offline results are not cached,
       so the next request tries the network again.

Complexity:
    cache lookup    O(1) average
    parsing         O(d * f) for d = 7 days and f = 7 fields -> constant
    network call    dominated by latency (~100 ms to a few s), not by CPU
    space           O(capacity) cached forecasts, each O(d)
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlencode

from leafcheck.http_json import fetch_json
from leafcheck.ttl_cache import TTLCache

API_URL: str = "https://api.open-meteo.com/v1/forecast"
DAILY_FIELDS: tuple[str, ...] = (
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
)
# Open-Meteo field name -> our short output key.
FIELD_KEYS: dict[str, str] = {
    "temperature_2m_max": "t_max",
    "temperature_2m_min": "t_min",
    "relative_humidity_2m_max": "humidity_max",
    "relative_humidity_2m_mean": "humidity_mean",
    "precipitation_sum": "rain_mm",
    "precipitation_probability_max": "rain_prob",
    "wind_speed_10m_max": "wind_max",
}
FORECAST_DAYS: int = 7
TIMEOUT_SECONDS: float = 6
OFFLINE_PATH: Path = Path(__file__).resolve().parent / "data" / "offline_forecast.json"

Fetcher = Callable[[str], Any]

# Live forecasts only; 64 locations, 30 minutes each.
_CACHE: TTLCache[tuple[float, float], dict] = TTLCache(capacity=64, ttl_seconds=1800)


def _fetch_json(url: str, timeout: float = TIMEOUT_SECONDS) -> Any:
    """Default fetcher: real HTTP GET with a User-Agent header."""
    return fetch_json(url, timeout=timeout)


def build_url(lat: float, lon: float) -> str:
    """Return the Open-Meteo request URL for (lat, lon). O(1)."""
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "daily": ",".join(DAILY_FIELDS),
        "timezone": "Asia/Kolkata",
        "forecast_days": FORECAST_DAYS,
    }
    # safe="," keeps the comma list readable; "/" in the timezone becomes %2F.
    return f"{API_URL}?{urlencode(params, safe=',')}"


def _round1(value: Any) -> Optional[float]:
    """Round a number to 1 decimal; None for missing or non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value):
        return None
    return round(float(value), 1)


def parse_forecast(payload: dict) -> list[dict]:
    """Turn Open-Meteo's column arrays into a list of per-day dicts. O(days).

    Raises:
        KeyError / TypeError / ValueError if the payload has no usable "daily" data.
    """
    daily = payload["daily"]
    dates = daily["time"]
    if not dates:
        raise ValueError("forecast has no days")
    days: list[dict] = []
    for i, day in enumerate(dates[:FORECAST_DAYS]):
        row: dict[str, Any] = {"date": str(day)}
        for field, key in FIELD_KEYS.items():
            column = daily.get(field) or []
            row[key] = _round1(column[i]) if i < len(column) else None
        days.append(row)
    return days


def load_offline_forecast(today: Optional[date] = None) -> dict:
    """Load the bundled sample forecast, dated from `today` onwards. O(days)."""
    start = today or date.today()
    with OFFLINE_PATH.open(encoding="utf-8") as f:
        sample = json.load(f)
    days = []
    for row in sample["days"]:
        day = {"date": (start + timedelta(days=row["day_offset"])).isoformat()}
        day.update({key: row.get(key) for key in FIELD_KEYS.values()})
        days.append(day)
    return {"source": "offline", "location": sample.get("location"), "days": days}


def get_forecast(lat: float, lon: float, *, fetch: Fetcher = _fetch_json) -> dict:
    """Return a 7-day forecast for (lat, lon).

    Returns:
        {"source": "live" | "offline",
         "days": [{"date": "YYYY-MM-DD", "t_max", "t_min", "humidity_max",
                   "humidity_mean", "rain_mm", "rain_prob", "wind_max"}, ...]}
        Numbers are floats rounded to 1 decimal, or None when missing.
    """
    key = (round(lat, 2), round(lon, 2))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        days = parse_forecast(fetch(build_url(lat, lon)))
    except Exception:  # any network, timeout, JSON or shape problem
        return load_offline_forecast()
    result = {"source": "live", "days": days}
    _CACHE.put(key, result)
    return result


def clear_cache() -> None:
    """Forget cached forecasts (used by tests)."""
    _CACHE.clear()
