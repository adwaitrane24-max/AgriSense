"""Farmer-friendly explanation of a predicted class, in English, Hindi or Marathi.

The text lives in leafcheck/data/advisory.json, shaped as
    { "<label>": { "en": {...}, "hi": {...}, "mr": {...} } }
where each language entry has: crop, condition, severity (none|low|medium|high,
always the English keyword), harm, economic_risk and exactly 3 next_steps.

It only describes the problem and safe next steps (check other plants, take
photos, contact the local Krishi Vigyan Kendra). It never gives treatment advice.

Complexity:
    load (once, at import)   O(file size) time and space
    get_advisory             O(1) average: two dict lookups, plus a copy of one
                             small fixed-size entry
    display_name             O(1) average for a known label
"""

import copy
import json
from pathlib import Path
from typing import Any, Optional

from leafcheck.predictor import pretty_label

SUPPORTED_LANGS: tuple[str, ...] = ("en", "hi", "mr")
DEFAULT_LANG: str = "en"
ADVISORY_PATH: Path = Path(__file__).resolve().parent / "data" / "advisory.json"


def _load(path: Path = ADVISORY_PATH) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the advisory JSON file once. O(file size) time and space."""
    return json.loads(path.read_text(encoding="utf-8"))


_ADVISORY: dict[str, dict[str, dict[str, Any]]] = _load()


def normalise_lang(lang: Optional[str]) -> str:
    """Return `lang` if we support it, otherwise English. O(1)."""
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def get_advisory(label: str, lang: str = DEFAULT_LANG) -> Optional[dict[str, Any]]:
    """Advisory entry for a class label in one language, or None if the label is unknown.

    Unsupported languages fall back to English. A deep copy is returned so a
    caller changing the result cannot change the shared data. O(1) average.
    """
    entry = _ADVISORY.get(label)
    if entry is None:
        return None
    language_entry = entry.get(normalise_lang(lang)) or entry[DEFAULT_LANG]
    return copy.deepcopy(language_entry)


def display_name(label: str, lang: str = DEFAULT_LANG) -> tuple[str, str]:
    """(crop, condition) to show on screen, translated when we have the label. O(1).

    Unknown labels fall back to the English names from predictor.pretty_label.
    """
    entry = _ADVISORY.get(label)
    if entry is None:
        return pretty_label(label)
    language_entry = entry.get(normalise_lang(lang)) or entry[DEFAULT_LANG]
    return language_entry["crop"], language_entry["condition"]
