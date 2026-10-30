"""Tests for the weather-based disease-risk outlook (no model or network needed)."""

import json
import re
from typing import Any, Optional

import pytest

from leafcheck import config
from leafcheck.disease_risk import (
    LEVELS,
    REASON_KEYS,
    RISK_TEXT_PATH,
    SUPPORTED_LANGS,
    TIP_KEYS,
    assess,
    field_tips,
    risk_text,
)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Same list as tests/test_advisory.py: no treatment words in any language.
FORBIDDEN_WORDS = (
    "spray", "pesticide", "fungicide", "insecticide", "bactericide", "chemical",
    "dose", "neem", "copper", "mancozeb", "remedy",
    "छिड़काव", "कीटनाशक", "फवारणी", "कीटकनाशक", "औषध",
)

DISEASE_LABELS = [label for label in config.CLASS_NAMES if not label.endswith("___healthy")]


def make_week(**values: Optional[float]) -> list[dict[str, Any]]:
    """Seven identical days with the given weather values."""
    base: dict[str, Any] = {
        "t_max": 28.0, "t_min": 20.0, "humidity_max": 70.0, "humidity_mean": 60.0,
        "rain_mm": 0.0, "rain_prob": 50, "wind_max": 10.0,
    }
    base.update(values)
    return [{"date": f"2026-11-0{i + 1}", **base} for i in range(7)]


def by_label(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["label"]: r for r in results}


def test_cool_wet_week_gives_high_late_blight() -> None:
    week = make_week(t_max=20, t_min=12, humidity_max=96, humidity_mean=88, rain_mm=8)
    results = by_label(assess(week))
    for label in ("Potato___Late_blight", "Tomato___Late_blight"):
        assert results[label]["level"] == "high"
        assert results[label]["reason"] == "cool_wet"
        assert results[label]["high_days"] == 7
        assert results[label]["peak_date"] == "2026-11-01"


def test_hot_dry_week_gives_high_mites_and_low_late_blight() -> None:
    week = make_week(t_max=37, t_min=24, humidity_max=45, humidity_mean=35, rain_mm=0)
    results = by_label(assess(week))
    assert results["Tomato___Spider_mites Two-spotted_spider_mite"]["level"] == "high"
    assert results["Potato___Late_blight"]["level"] == "low"
    assert results["Potato___Late_blight"]["peak_date"] is None
    assert results["Tomato___Late_blight"]["level"] == "low"


def test_two_favourable_days_gives_medium() -> None:
    week = make_week(t_max=37, t_min=24, humidity_mean=60, rain_mm=0)
    for day in week[:2]:
        day["humidity_mean"] = 40
    results = by_label(assess(week))
    mites = results["Tomato___Spider_mites Two-spotted_spider_mite"]
    assert (mites["level"], mites["high_days"]) == ("medium", 2)


def test_every_disease_label_once_and_no_healthy() -> None:
    results = assess(make_week())
    labels = [r["label"] for r in results]
    assert sorted(labels) == sorted(DISEASE_LABELS)
    assert len(labels) == len(set(labels))
    assert not any(label.endswith("___healthy") for label in labels)
    for r in results:
        assert r["level"] in LEVELS
        assert r["crop"] in {"tomato", "potato", "corn", "pepper"}
        assert r["reason"] in REASON_KEYS


def test_results_sorted_by_level_then_crop() -> None:
    week = make_week(t_max=20, t_min=12, humidity_max=96, humidity_mean=88, rain_mm=8)
    results = assess(week)
    rank = {"high": 0, "medium": 1, "low": 2}
    keys = [(rank[r["level"]], r["crop"]) for r in results]
    assert keys == sorted(keys)
    assert results[0]["level"] == "high"


def test_mosaic_virus_is_never_weather_driven() -> None:
    week = make_week(t_max=20, t_min=12, humidity_max=96, humidity_mean=88, rain_mm=8)
    mosaic = by_label(assess(week))["Tomato___Tomato_mosaic_virus"]
    assert (mosaic["level"], mosaic["reason"]) == ("low", "not_weather_driven")


def test_missing_values_do_not_crash() -> None:
    week = make_week(t_max=None, humidity_mean=None, rain_mm=None, rain_prob=None, wind_max=None)
    week.append({"date": "2026-11-08"})  # a day with nothing at all
    results = assess(week)
    assert len(results) == len(DISEASE_LABELS)
    assert all(r["level"] == "low" for r in results)
    assert field_tips(week)[-1]["key"] == "contact_kvk"


def test_empty_forecast_gives_all_low() -> None:
    results = assess([])
    assert len(results) == len(DISEASE_LABELS)
    assert all(r["level"] == "low" and r["high_days"] == 0 for r in results)
    assert field_tips([]) == [{"key": "contact_kvk", "date": None}]


def test_heavy_rain_tip_has_its_date() -> None:
    week = make_week(rain_mm=0)
    week[3]["rain_mm"] = 45
    tips = field_tips(week)
    heavy = [t for t in tips if t["key"] == "heavy_rain"]
    assert heavy == [{"key": "heavy_rain", "date": "2026-11-04"}]
    assert tips[-1] == {"key": "contact_kvk", "date": None}


def test_tips_capped_and_end_with_kvk() -> None:
    week = make_week(t_max=40, rain_mm=25, humidity_max=95, wind_max=50, rain_prob=10)
    tips = field_tips(week)
    assert len(tips) <= 6
    assert tips[-1]["key"] == "contact_kvk"
    assert all(t["key"] in TIP_KEYS for t in tips)


def test_dry_spell_and_photo_day_tips() -> None:
    week = make_week(rain_mm=0, humidity_mean=40, rain_prob=10)
    keys = [t["key"] for t in field_tips(week)]
    assert "dry_spell" in keys and "good_photo_day" in keys


@pytest.mark.parametrize(
    ("kind", "key"),
    [("reason", k) for k in REASON_KEYS] + [("tip", k) for k in TIP_KEYS] + [("level", k) for k in LEVELS],
)
def test_every_key_has_text_in_all_languages(kind: str, key: str) -> None:
    english = risk_text(kind, key, "en")
    assert english
    for lang in SUPPORTED_LANGS:
        text = risk_text(kind, key, lang)
        assert text
        if lang in ("hi", "mr"):
            assert DEVANAGARI.search(text), f"{kind}/{key}/{lang}: {text!r}"
        assert ("{date}" in text) == ("{date}" in english)


def test_unknown_language_falls_back_to_english() -> None:
    assert risk_text("level", "high", "fr") == "High"


def test_risk_text_has_no_treatment_words() -> None:
    text = RISK_TEXT_PATH.read_text(encoding="utf-8").lower()
    json.loads(text)  # still valid JSON
    for word in FORBIDDEN_WORDS:
        assert word not in text, f"risk_text.json must not mention {word!r}"
