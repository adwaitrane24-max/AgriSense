"""Tests for the Maharashtra district table and lookups."""

import re

from leafcheck.districts import (
    DISTRICTS,
    SOIL_TYPES,
    get_district,
    haversine_km,
    in_maharashtra,
    nearest_district,
)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def test_has_36_districts_with_unique_slugs() -> None:
    assert len(DISTRICTS) == 36
    assert len({d.slug for d in DISTRICTS}) == 36


def test_every_district_is_inside_bounding_box() -> None:
    for d in DISTRICTS:
        assert in_maharashtra(d.lat, d.lon), d.slug


def test_names_in_three_languages_with_devanagari() -> None:
    for d in DISTRICTS:
        assert set(d.name) == {"en", "hi", "mr"}
        assert d.name["en"].isascii()
        assert DEVANAGARI.search(d.name["hi"]), d.slug
        assert DEVANAGARI.search(d.name["mr"]), d.slug


def test_every_soil_key_has_text_in_all_languages() -> None:
    for d in DISTRICTS:
        assert d.soil in SOIL_TYPES, d.slug
    for key, langs in SOIL_TYPES.items():
        assert set(langs) == {"en", "hi", "mr"}, key
        for text in langs.values():
            assert text["name"] and text["description"]


def test_nearest_district_pune_and_nagpur() -> None:
    assert nearest_district(18.52, 73.86).slug == "pune"
    assert nearest_district(21.15, 79.09).slug == "nagpur"
    # A point a little outside the HQ still maps to the same district.
    assert nearest_district(18.60, 73.80).slug == "pune"


def test_in_maharashtra_rejects_far_points() -> None:
    assert not in_maharashtra(28.61, 77.21)  # Delhi
    assert not in_maharashtra(12.97, 77.59)  # Bengaluru


def test_get_district() -> None:
    pune = get_district("pune")
    assert pune is not None and pune.name["mr"] == "पुणे"
    assert get_district("atlantis") is None


def test_haversine_known_distance() -> None:
    # Pune to Mumbai is roughly 120 km in a straight line.
    assert 110 < haversine_km(18.52, 73.86, 18.94, 72.83) < 130
    assert haversine_km(18.52, 73.86, 18.52, 73.86) == 0
