"""Weather-based disease-risk outlook for the next 7 days.

Many leaf diseases need particular weather to spread: late blight likes cool,
wet days; spider mites like hot, dry days. This module reads a 7-day forecast
and says, for each of the 15 disease classes LeafCheck knows, whether the
coming week looks favourable (low / medium / high risk). It is a simple,
explainable rule table, NOT a validated forecasting model, and it gives no
treatment advice: the output only tells a farmer when to look at leaves more
closely and to confirm with a local expert.

Input: a list of day dicts, one per day::

    {"date": "YYYY-MM-DD", "t_max": float|None, "t_min": float|None,
     "humidity_max": float|None, "humidity_mean": float|None,
     "rain_mm": float|None, "rain_prob": int|None, "wind_max": float|None}

Units: degrees Celsius, percent, millimetres, km/h. t_mean = (t_max + t_min) / 2.

Rule table (a day is "favourable" when every condition holds; a day with a
missing value that the rule needs is skipped, i.e. not favourable):

| Disease (crop)                         | Condition on one day                                      | reason key         |
| -------------------------------------- | --------------------------------------------------------- | ------------------ |
| Late blight (potato, tomato)           | 10 <= t_mean <= 24 and (humidity_max >= 90 or rain >= 2)  | cool_wet           |
| Early blight (potato, tomato)          | 20 <= t_mean <= 30 and (humidity_mean >= 75 or rain >= 2) | warm_humid         |
| Septoria leaf spot, Target spot (tom.) | 20 <= t_mean <= 30 and (humidity_mean >= 75 or rain >= 2) | warm_humid         |
| Bacterial spot (tomato, pepper)        | 22 <= t_mean <= 32 and rain >= 5                          | warm_rain_splash   |
| Leaf mold (tomato)                     | 18 <= t_mean <= 27 and humidity_mean >= 85                | very_humid         |
| Spider mites (tomato)                  | t_max >= 30 and humidity_mean <= 50 and rain < 1          | hot_dry            |
| Yellow leaf curl virus (tomato)        | t_mean >= 25 and humidity_mean <= 65 and rain < 1         | vector_weather     |
| Mosaic virus (tomato)                  | never (spread by handling and seed, not weather)          | not_weather_driven |
| Common rust (corn)                     | 16 <= t_mean <= 25 and (humidity_max >= 95 or rain >= 2)  | cool_humid         |
| Northern leaf blight (corn)            | 18 <= t_mean <= 27 and (humidity_max >= 95 or rain >= 2)  | mild_wet           |
| Gray leaf spot (corn)                  | 22 <= t_mean <= 30 and humidity_mean >= 85                | warm_very_humid    |

Level: "high" if 3 or more favourable days, "medium" if 1 or 2, else "low".

Complexity: assess() is O(d x r) time for d days and r disease rules (both
small constants: d = 7, r = 15) and O(r) extra space. field_tips() is O(d).
risk_text() is an O(1) dictionary lookup after the JSON is loaded once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Optional

from leafcheck import config

RISK_TEXT_PATH: Path = Path(__file__).resolve().parent / "data" / "risk_text.json"
SUPPORTED_LANGS: tuple[str, ...] = ("en", "hi", "mr")
DEFAULT_LANG: str = "en"

LEVELS: tuple[str, ...] = ("low", "medium", "high")
HIGH_MIN_DAYS: int = 3  # favourable days needed for "high"
MEDIUM_MIN_DAYS: int = 1  # favourable days needed for "medium"
MAX_TIPS: int = 5  # weather tips before the final "contact_kvk" tip

Day = dict[str, Any]


# ---------------------------------------------------------------------------
# Small helpers: each reads one day and returns a number or None.
# ---------------------------------------------------------------------------

def _t_mean(day: Day) -> Optional[float]:
    """Mean temperature of a day, or None if either extreme is missing. O(1)."""
    t_max, t_min = day.get("t_max"), day.get("t_min")
    if t_max is None or t_min is None:
        return None
    return (t_max + t_min) / 2


def _all_present(day: Day, *keys: str) -> bool:
    """True when every named value is present (not None). O(number of keys)."""
    return all(day.get(key) is not None for key in keys)


# ---------------------------------------------------------------------------
# Day conditions, one per rule family (see the table in the module docstring).
# Each returns False when a value it needs is missing, so the day is skipped.
# ---------------------------------------------------------------------------

def _cool_wet(day: Day) -> bool:
    """Late blight: cool and very humid or rainy. O(1)."""
    t = _t_mean(day)
    if t is None or not _all_present(day, "humidity_max", "rain_mm"):
        return False
    return 10 <= t <= 24 and (day["humidity_max"] >= 90 or day["rain_mm"] >= 2)


def _warm_humid(day: Day) -> bool:
    """Early blight, Septoria, target spot: warm and humid or rainy. O(1)."""
    t = _t_mean(day)
    if t is None or not _all_present(day, "humidity_mean", "rain_mm"):
        return False
    return 20 <= t <= 30 and (day["humidity_mean"] >= 75 or day["rain_mm"] >= 2)


def _warm_rain_splash(day: Day) -> bool:
    """Bacterial spot: warm with heavy rain that splashes bacteria. O(1)."""
    t = _t_mean(day)
    if t is None or day.get("rain_mm") is None:
        return False
    return 22 <= t <= 32 and day["rain_mm"] >= 5


def _very_humid(day: Day) -> bool:
    """Leaf mold: mild and very humid all day. O(1)."""
    t = _t_mean(day)
    if t is None or day.get("humidity_mean") is None:
        return False
    return 18 <= t <= 27 and day["humidity_mean"] >= 85


def _hot_dry(day: Day) -> bool:
    """Spider mites: hot, dry air and little rain. O(1)."""
    if not _all_present(day, "t_max", "humidity_mean", "rain_mm"):
        return False
    return day["t_max"] >= 30 and day["humidity_mean"] <= 50 and day["rain_mm"] < 1


def _vector_weather(day: Day) -> bool:
    """Leaf curl virus: warm, dry spells that suit whiteflies (the vector). O(1)."""
    t = _t_mean(day)
    if t is None or not _all_present(day, "humidity_mean", "rain_mm"):
        return False
    return t >= 25 and day["humidity_mean"] <= 65 and day["rain_mm"] < 1


def _never(day: Day) -> bool:
    """Mosaic virus: weather is not the driver, so no day counts. O(1)."""
    return False


def _cool_humid(day: Day) -> bool:
    """Common rust (corn): cool to moderate and very humid or rainy. O(1)."""
    t = _t_mean(day)
    if t is None or not _all_present(day, "humidity_max", "rain_mm"):
        return False
    return 16 <= t <= 25 and (day["humidity_max"] >= 95 or day["rain_mm"] >= 2)


def _mild_wet(day: Day) -> bool:
    """Northern leaf blight (corn): mild and wet. O(1)."""
    t = _t_mean(day)
    if t is None or not _all_present(day, "humidity_max", "rain_mm"):
        return False
    return 18 <= t <= 27 and (day["humidity_max"] >= 95 or day["rain_mm"] >= 2)


def _warm_very_humid(day: Day) -> bool:
    """Gray leaf spot (corn): warm and very humid. O(1)."""
    t = _t_mean(day)
    if t is None or day.get("humidity_mean") is None:
        return False
    return 22 <= t <= 30 and day["humidity_mean"] >= 85


@dataclass(frozen=True)
class Rule:
    """One disease's weather rule: which days favour it and why."""

    reason: str
    favourable: Callable[[Day], bool]


# Disease label -> rule. Healthy labels are deliberately absent.
RULES: dict[str, Rule] = {
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": Rule("warm_very_humid", _warm_very_humid),
    "Corn___Common_rust": Rule("cool_humid", _cool_humid),
    "Corn___Northern_Leaf_Blight": Rule("mild_wet", _mild_wet),
    "Pepper,_bell___Bacterial_spot": Rule("warm_rain_splash", _warm_rain_splash),
    "Potato___Early_blight": Rule("warm_humid", _warm_humid),
    "Potato___Late_blight": Rule("cool_wet", _cool_wet),
    "Tomato___Bacterial_spot": Rule("warm_rain_splash", _warm_rain_splash),
    "Tomato___Early_blight": Rule("warm_humid", _warm_humid),
    "Tomato___Late_blight": Rule("cool_wet", _cool_wet),
    "Tomato___Leaf_Mold": Rule("very_humid", _very_humid),
    "Tomato___Septoria_leaf_spot": Rule("warm_humid", _warm_humid),
    "Tomato___Spider_mites Two-spotted_spider_mite": Rule("hot_dry", _hot_dry),
    "Tomato___Target_Spot": Rule("warm_humid", _warm_humid),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": Rule("vector_weather", _vector_weather),
    "Tomato___Tomato_mosaic_virus": Rule("not_weather_driven", _never),
}

_CROP_PREFIXES: dict[str, str] = {
    "Corn": "corn",
    "Pepper,_bell": "pepper",
    "Potato": "potato",
    "Tomato": "tomato",
}

REASON_KEYS: tuple[str, ...] = tuple(sorted({rule.reason for rule in RULES.values()}))
TIP_KEYS: tuple[str, ...] = (
    "heavy_rain", "high_humidity_run", "heat", "strong_wind", "dry_spell",
    "good_photo_day", "contact_kvk",
)


def crop_of(label: str) -> str:
    """Short crop name ("tomato", "potato", "corn", "pepper") from a label. O(1)."""
    return _CROP_PREFIXES[label.split("___", 1)[0]]


def is_healthy(label: str) -> bool:
    """True for the four "healthy" classes, which have no risk rule. O(1)."""
    return label.endswith("___healthy")


def level_for(high_days: int) -> str:
    """Map a count of favourable days to "low", "medium" or "high". O(1)."""
    if high_days >= HIGH_MIN_DAYS:
        return "high"
    if high_days >= MEDIUM_MIN_DAYS:
        return "medium"
    return "low"


def _assess_one(label: str, rule: Rule, days: list[Day]) -> dict[str, Any]:
    """Score one disease over all days. O(d) time, O(1) extra space.

    The daily score is 1 for a favourable day and 0 otherwise; ``high_days``
    is their sum and ``peak_date`` is the first favourable day (None if none).
    """
    high_days = 0
    peak_date: Optional[str] = None
    for day in days:
        if rule.favourable(day):
            high_days += 1
            if peak_date is None:
                peak_date = day.get("date")
    return {
        "label": label,
        "crop": crop_of(label),
        "level": level_for(high_days),
        "peak_date": peak_date,
        "high_days": high_days,
        "reason": rule.reason,
    }


def assess(days: list[Day]) -> list[dict[str, Any]]:
    """Weekly risk level for every disease class (healthy classes skipped).

    Returns one dict per disease label in ``config.CLASS_NAMES``::

        {"label", "crop", "level", "peak_date", "high_days", "reason"}

    sorted by level (high first), then crop, then label.

    Time O(d x r) for d days and r disease rules (d = 7, r = 15); sorting r
    items adds O(r log r). Space O(r) for the output list.
    """
    results = [
        _assess_one(label, RULES[label], days)
        for label in config.CLASS_NAMES
        if not is_healthy(label)
    ]
    rank = {level: index for index, level in enumerate(LEVELS)}
    results.sort(key=lambda r: (-rank[r["level"]], r["crop"], r["label"]))
    return results


# ---------------------------------------------------------------------------
# Field tips: safe "what to watch for" notes, never treatment advice.
# ---------------------------------------------------------------------------

def _first_date(days: list[Day], test: Callable[[Day], bool]) -> Optional[str]:
    """Date of the first day passing ``test``, or None. O(d)."""
    for day in days:
        if test(day):
            return day.get("date")
    return None


def _run_start(days: list[Day], test: Callable[[Day], bool], length: int) -> Optional[str]:
    """Start date of the first run of ``length`` consecutive days passing ``test``.

    One pass with a running counter: O(d) time, O(1) space.
    """
    run = 0
    for index, day in enumerate(days):
        run = run + 1 if test(day) else 0
        if run == length:
            return days[index - length + 1].get("date")
    return None


def _ge(key: str, limit: float) -> Callable[[Day], bool]:
    """Build a test "value of key is present and >= limit". O(1)."""
    return lambda day: day.get(key) is not None and day[key] >= limit


def _is_dry(day: Day) -> bool:
    """No real rain and fairly dry air. O(1)."""
    return (
        _all_present(day, "rain_mm", "humidity_mean")
        and day["rain_mm"] < 1
        and day["humidity_mean"] <= 60
    )


def _is_clear(day: Day) -> bool:
    """Low chance of rain, so daylight is good for leaf photos. O(1)."""
    return day.get("rain_prob") is not None and day["rain_prob"] < 30


def field_tips(days: list[Day]) -> list[dict[str, Optional[str]]]:
    """Up to 5 weather-based observation tips, then always "contact_kvk".

    Each tip is ``{"key": tip_key, "date": "YYYY-MM-DD" or None}``; the text
    comes from ``risk_text("tip", key, lang)``. Tips only say what to look
    for and when; they never suggest a treatment.

    Time O(d) (a fixed number of passes over the days), space O(1).
    """
    candidates: list[tuple[str, Optional[str]]] = [
        ("heavy_rain", _first_date(days, _ge("rain_mm", 20))),
        ("high_humidity_run", _run_start(days, _ge("humidity_max", 90), 3)),
        ("heat", _first_date(days, _ge("t_max", 38))),
        ("strong_wind", _first_date(days, _ge("wind_max", 35))),
        ("dry_spell", _run_start(days, _is_dry, 5)),
        ("good_photo_day", _first_date(days, _is_clear)),
    ]
    tips = [{"key": key, "date": date} for key, date in candidates if date is not None]
    tips = tips[:MAX_TIPS]
    tips.append({"key": "contact_kvk", "date": None})
    return tips


# ---------------------------------------------------------------------------
# Text in English, Hindi and Marathi.
# ---------------------------------------------------------------------------

@cache
def _load_text(path: Path = RISK_TEXT_PATH) -> dict[str, dict[str, dict[str, str]]]:
    """Read risk_text.json once and keep it in memory. O(file size) once."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalise_lang(lang: Optional[str]) -> str:
    """Return ``lang`` if supported, else English. O(1)."""
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def risk_text(kind: str, key: str, lang: str = DEFAULT_LANG) -> str:
    """Text for a "reason", "tip" or "level" key in en / hi / mr.

    Falls back to English when the language is unknown or missing. Raises
    KeyError for an unknown kind or key. Tip text may contain "{date}".
    O(1) dictionary lookups.
    """
    entry = _load_text()[kind][key]
    return entry.get(normalise_lang(lang)) or entry[DEFAULT_LANG]
