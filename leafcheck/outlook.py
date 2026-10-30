"""Combine weather, soil and disease-risk data into one Field Outlook response.

    location -> nearest district (O(36))
    weather  -> 7-day forecast (TTL-cached, offline fallback)
    soil     -> SoilGrids estimate (TTL-cached) or the district's general soil type
    risks    -> weather-based disease risk for our 15 disease classes
    tips     -> safe field tips

Everything returned is already translated into the requested language.
Complexity: O(d * r) for d = 7 days and r = 15 risk rules; network calls dominate
on a cache miss, a cache hit is O(1).
"""

from typing import Any

from leafcheck import disease_risk, districts, soil, weather
from leafcheck.advisory import display_name, normalise_lang
from leafcheck.i18n import strings_for


def _field(item: Any, key: str) -> Any:
    """Read `key` from a dict or a dataclass-like district record."""
    return item[key] if isinstance(item, dict) else getattr(item, key)


def district_options(lang: str) -> list[dict[str, Any]]:
    """All districts as {slug, name, lat, lon} for the map and dropdown, sorted by name."""
    lang = normalise_lang(lang)
    options = [
        {"slug": _field(d, "slug"), "name": _field(d, "name").get(lang) or _field(d, "name")["en"],
         "lat": _field(d, "lat"), "lon": _field(d, "lon")}
        for d in districts.DISTRICTS
    ]
    return sorted(options, key=lambda o: o["name"])


def _soil_block(raw: dict[str, Any], district: Any, lang: str) -> dict[str, Any]:
    """Soil values plus translated soil-type and texture names."""
    soil_key = raw.get("soil") or _field(district, "soil")
    soil_type = districts.SOIL_TYPES.get(soil_key, {})
    type_text = soil_type.get(lang) or soil_type.get("en") or {}
    texture = raw.get("texture")
    texture_names = soil.TEXTURE_NAMES.get(texture, {}) if texture else {}
    return {
        "source": raw.get("source", "district"),
        "ph": raw.get("ph"),
        "clay_pct": raw.get("clay_pct"),
        "sand_pct": raw.get("sand_pct"),
        "silt_pct": raw.get("silt_pct"),
        "organic_carbon_gkg": raw.get("organic_carbon_gkg"),
        "texture_name": texture_names.get(lang) or texture_names.get("en"),
        "type_name": type_text.get("name"),
        "type_description": type_text.get("description"),
    }


def _risk_block(entry: dict[str, Any], lang: str) -> dict[str, Any]:
    crop, condition = display_name(entry["label"], lang)
    peak = entry.get("peak_date")
    return {
        "label": entry["label"],
        "crop_name": crop,
        "disease_name": condition,
        "level": entry["level"],
        "level_class": entry["level"],
        "level_text": disease_risk.risk_text("level", entry["level"], lang),
        "reason_text": disease_risk.risk_text("reason", entry["reason"], lang),
        "peak_date": peak,
        "peak_text": strings_for(lang)["risk_peak"].replace("{date}", peak) if peak else None,
        "high_days": entry["high_days"],
    }


def _tip_block(tip: dict[str, Any], lang: str) -> dict[str, Any]:
    text = disease_risk.risk_text("tip", tip["key"], lang)
    return {"key": tip["key"], "date": tip.get("date"), "text": text.replace("{date}", tip.get("date") or "")}


def build_outlook(lat: float, lon: float, lang: str | None) -> dict[str, Any]:
    """Full outlook for a point. Caller must check `districts.in_maharashtra` first."""
    lang = normalise_lang(lang)
    district = districts.nearest_district(lat, lon)
    forecast = weather.get_forecast(lat, lon)
    days = forecast["days"]
    names = _field(district, "name")
    return {
        "lat": lat,
        "lon": lon,
        "lang": lang,
        "district": {"slug": _field(district, "slug"), "name": names.get(lang) or names["en"]},
        "weather": forecast,
        "soil": _soil_block(soil.get_soil(lat, lon), district, lang),
        "risks": [_risk_block(r, lang) for r in disease_risk.assess(days)],
        "tips": [_tip_block(t, lang) for t in disease_risk.field_tips(days)],
    }
