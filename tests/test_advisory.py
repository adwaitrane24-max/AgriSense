"""Tests for the multilingual farmer advisory data and lookup functions."""

import json
import re

import pytest

from leafcheck import config
from leafcheck.advisory import (
    ADVISORY_PATH,
    SUPPORTED_LANGS,
    display_name,
    get_advisory,
    normalise_lang,
)
from leafcheck.predictor import pretty_label

LABELS = list(config.CLASS_NAMES)
KEYS = {"crop", "condition", "severity", "harm", "economic_risk", "next_steps"}
SEVERITIES = {"none", "low", "medium", "high"}
DEVANAGARI = re.compile("[ऀ-ॿ]")

FORBIDDEN_WORDS = (
    "spray", "pesticide", "fungicide", "insecticide", "bactericide", "chemical",
    "dose", "neem", "copper", "mancozeb", "remedy",
    "छिड़काव", "कीटनाशक", "फवारणी", "कीटकनाशक", "औषध",
)


@pytest.mark.parametrize("label", LABELS)
def test_every_label_has_complete_entries(label: str) -> None:
    severities = set()
    for lang in SUPPORTED_LANGS:
        entry = get_advisory(label, lang)
        assert entry is not None
        assert set(entry) == KEYS
        for key in ("crop", "condition", "harm", "economic_risk"):
            assert isinstance(entry[key], str) and entry[key].strip()
        assert len(entry["next_steps"]) == 3
        assert all(isinstance(step, str) and step.strip() for step in entry["next_steps"])
        assert entry["severity"] in SEVERITIES
        severities.add(entry["severity"])
    assert len(severities) == 1  # severity is a code, same in every language


def test_json_has_exactly_the_19_labels() -> None:
    data = json.loads(ADVISORY_PATH.read_text(encoding="utf-8"))
    assert set(data) == set(LABELS)


@pytest.mark.parametrize("label", [name for name in LABELS if name.endswith("healthy")])
def test_healthy_labels_have_no_severity(label: str) -> None:
    assert get_advisory(label)["severity"] == "none"


def test_diseased_labels_are_not_marked_none() -> None:
    for label in LABELS:
        if not label.endswith("healthy"):
            assert get_advisory(label)["severity"] != "none"


@pytest.mark.parametrize("lang,expected", [("hi", "hi"), ("mr", "mr"), ("fr", "en"), (None, "en"), ("", "en")])
def test_normalise_lang(lang: str | None, expected: str) -> None:
    assert normalise_lang(lang) == expected


def test_unknown_lang_falls_back_to_english() -> None:
    assert get_advisory("Tomato___Early_blight", "fr") == get_advisory("Tomato___Early_blight", "en")


def test_unknown_label_returns_none() -> None:
    assert get_advisory("Banana___Black_sigatoka") is None


def test_get_advisory_returns_a_copy() -> None:
    first = get_advisory("Potato___Late_blight", "hi")
    first["severity"] = "changed"
    first["next_steps"].append("extra")
    second = get_advisory("Potato___Late_blight", "hi")
    assert second["severity"] == "high"
    assert len(second["next_steps"]) == 3


def test_display_name_in_each_language() -> None:
    assert display_name("Tomato___Early_blight", "en") == ("Tomato", "Early blight")
    crop_hi, condition_hi = display_name("Tomato___Early_blight", "hi")
    crop_mr, condition_mr = display_name("Tomato___Early_blight", "mr")
    assert crop_hi == "टमाटर" and DEVANAGARI.search(condition_hi)
    assert crop_mr == "टोमॅटो" and DEVANAGARI.search(condition_mr)


def test_display_name_unknown_label_uses_pretty_label() -> None:
    assert display_name("Banana___Black_sigatoka", "hi") == pretty_label("Banana___Black_sigatoka")


def test_no_treatment_advice_anywhere() -> None:
    text = ADVISORY_PATH.read_text(encoding="utf-8").lower()
    for word in FORBIDDEN_WORDS:
        assert word not in text, f"advisory.json must not mention {word!r}"


@pytest.mark.parametrize("label", LABELS)
def test_hindi_and_marathi_use_devanagari(label: str) -> None:
    for lang in ("hi", "mr"):
        entry = get_advisory(label, lang)
        for text in [entry["crop"], entry["condition"], entry["harm"], entry["economic_risk"], *entry["next_steps"]]:
            assert DEVANAGARI.search(text), f"{label}/{lang}: {text!r}"
