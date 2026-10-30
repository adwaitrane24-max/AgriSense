"""Tests for the English / Hindi / Marathi UI strings and the ?lang= switch."""

import re

import pytest
from flask.testing import FlaskClient

from leafcheck.i18n import (
    LANG_NAMES,
    SUPPORTED_LANGS,
    UI_STRINGS,
    normalise_lang,
    strings_for,
)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
TREATMENT_WORDS = ("spray", "pesticide", "fungicide", "छिड़काव", "कीटनाशक", "फवारणी")


@pytest.mark.parametrize("lang", SUPPORTED_LANGS)
def test_every_language_has_every_english_key(lang: str) -> None:
    assert set(UI_STRINGS[lang]) >= set(UI_STRINGS["en"])


@pytest.mark.parametrize("lang", SUPPORTED_LANGS)
def test_values_are_non_empty(lang: str) -> None:
    assert all(value.strip() for value in UI_STRINGS[lang].values())


@pytest.mark.parametrize("lang", ["hi", "mr"])
def test_hindi_and_marathi_use_devanagari(lang: str) -> None:
    for key in ("btn_analyse", "disclaimer", "advisory_title", "not_sure_message"):
        assert DEVANAGARI.search(UI_STRINGS[lang][key]), key


@pytest.mark.parametrize("lang", SUPPORTED_LANGS)
def test_placeholders_match_english(lang: str) -> None:
    for key, english in UI_STRINGS["en"].items():
        assert re.findall(r"\{\w+\}", UI_STRINGS[lang][key]) == re.findall(r"\{\w+\}", english), key


def test_no_treatment_words_in_any_language() -> None:
    for lang, table in UI_STRINGS.items():
        for key, value in table.items():
            for word in TREATMENT_WORDS:
                assert word not in value.lower(), f"{lang}.{key} contains {word!r}"


def test_lang_names_cover_supported_langs() -> None:
    assert set(LANG_NAMES) == set(SUPPORTED_LANGS)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("hi", "hi"), ("MR", "mr"), (" en ", "en"), ("xx", "en"), ("", "en"), (None, "en")],
)
def test_normalise_lang(raw: str | None, expected: str) -> None:
    assert normalise_lang(raw) == expected


def test_strings_for_falls_back_to_english_for_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    partial = dict(UI_STRINGS["hi"])
    del partial["nav_stats"]
    monkeypatch.setitem(UI_STRINGS, "hi", partial)
    strings = strings_for("hi")
    assert strings["nav_stats"] == UI_STRINGS["en"]["nav_stats"]
    assert strings["btn_analyse"] == partial["btn_analyse"]


def test_strings_for_unknown_language_is_english() -> None:
    assert strings_for("xx") == UI_STRINGS["en"]


def test_home_in_hindi(client: FlaskClient) -> None:
    response = client.get("/?lang=hi")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert '<html lang="hi">' in page
    assert UI_STRINGS["hi"]["btn_analyse"] in page
    assert UI_STRINGS["hi"]["disclaimer"] in page
    assert 'id="i18n"' in page
    assert 'class="lang-switch"' in page


def test_unknown_language_falls_back_to_english(client: FlaskClient) -> None:
    page = client.get("/?lang=xx").get_data(as_text=True)
    assert '<html lang="en">' in page
    assert UI_STRINGS["en"]["btn_analyse"] in page


def test_stats_in_marathi_keeps_language_in_nav(client: FlaskClient) -> None:
    response = client.get("/stats?lang=mr")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert '<html lang="mr">' in page
    assert UI_STRINGS["mr"]["stats_hit_rate"] in page
    assert 'href="/?lang=mr"' in page  # the Analyse link keeps the language


def test_switcher_links_stay_on_current_page(client: FlaskClient) -> None:
    page = client.get("/stats?lang=hi").get_data(as_text=True)
    for code in SUPPORTED_LANGS:
        assert f'href="/stats?lang={code}"' in page
    assert 'aria-current="true">हिंदी</a>' in page
