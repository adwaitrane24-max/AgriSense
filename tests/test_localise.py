"""Tests for adding translated names and farmer guidance to results."""

import io

from flask.testing import FlaskClient

from leafcheck import config
from leafcheck.localise import localise_result
from leafcheck.predictor import build_result
from tests.conftest import confident_probabilities
from tests.helpers import make_image_bytes

LABELS = list(config.CLASS_NAMES)


def upload(client: FlaskClient, data: bytes, lang: str | None = None):
    form = {"image": (io.BytesIO(data), "leaf.jpg")}
    if lang is not None:
        form["lang"] = lang
    return client.post("/predict", data=form, content_type="multipart/form-data")


def test_confident_result_gets_advisory_and_names() -> None:
    result = localise_result(build_result(confident_probabilities(), LABELS), "hi")
    assert result["lang"] == "hi"
    assert result["advisory"] is not None
    assert len(result["advisory"]["next_steps"]) == 3
    assert result["prediction"]["display_crop"]
    assert all("display_condition" in item for item in result["top"])


def test_not_sure_result_has_no_advisory() -> None:
    result = localise_result(build_result(confident_probabilities(best=0.3), LABELS), "mr")
    assert result["status"] == "not_sure"
    assert result["advisory"] is None
    assert result["prediction"] is None


def test_unknown_language_falls_back_to_english() -> None:
    result = localise_result(build_result(confident_probabilities(), LABELS), "xx")
    assert result["lang"] == "en"


def test_input_result_is_not_modified() -> None:
    original = build_result(confident_probabilities(), LABELS)
    localise_result(original, "mr")
    assert "advisory" not in original
    assert "display_crop" not in original["top"][0]


def test_predict_returns_advisory_in_requested_language(client: FlaskClient) -> None:
    body = upload(client, make_image_bytes(), lang="mr").get_json()
    assert body["lang"] == "mr"
    assert body["advisory"]["severity"] in {"none", "low", "medium", "high"}


def test_cache_hit_can_be_served_in_another_language(client: FlaskClient) -> None:
    image = make_image_bytes()
    first = upload(client, image, lang="en").get_json()
    second = upload(client, image, lang="hi").get_json()
    assert second["from_cache"] is True
    assert second["lang"] == "hi"
    assert second["advisory"]["harm"] != first["advisory"]["harm"]


def test_predict_without_lang_defaults_to_english(client: FlaskClient) -> None:
    assert upload(client, make_image_bytes()).get_json()["lang"] == "en"
