"""End-to-end HTTP tests using the Flask test client and a fake predictor."""

import io

from flask.testing import FlaskClient

from leafcheck import config
from leafcheck.predictor import FakePredictor
from leafcheck.service import LeafCheckService
from tests.conftest import confident_probabilities
from tests.helpers import make_image_bytes


def upload(client: FlaskClient, data: bytes, filename: str = "leaf.jpg"):
    return client.post(
        "/predict",
        data={"image": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_home_page_has_form_and_disclaimer(client: FlaskClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'id="upload-form"' in page
    assert config.DISCLAIMER in page


def test_valid_image_returns_result(client: FlaskClient) -> None:
    response = upload(client, make_image_bytes())
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["prediction"]["label"] == "Tomato___Early_blight"
    assert len(body["top"]) == 3
    assert body["from_cache"] is False
    assert body["disclaimer"] == config.DISCLAIMER


def test_missing_file_field_returns_400(client: FlaskClient) -> None:
    response = client.post("/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "choose an image" in response.get_json()["error"]


def test_empty_upload_returns_400(client: FlaskClient) -> None:
    response = upload(client, b"")
    assert response.status_code == 400
    assert "empty" in response.get_json()["error"]


def test_text_file_returns_400(client: FlaskClient) -> None:
    response = upload(client, b"just some text", filename="notes.txt")
    assert response.status_code == 400
    assert "JPG and PNG" in response.get_json()["error"]


def test_oversize_upload_is_rejected(client: FlaskClient) -> None:
    huge = b"\xff\xd8\xff" + b"0" * (config.MAX_UPLOAD_BYTES + 10)
    response = upload(client, huge)
    assert response.status_code in (400, 413)
    assert "too large" in response.get_json()["error"]


def test_same_image_twice_comes_from_cache(
    client: FlaskClient, fake_predictor: FakePredictor
) -> None:
    image = make_image_bytes()
    first = upload(client, image).get_json()
    second = upload(client, image, filename="renamed.jpg").get_json()
    assert first["from_cache"] is False
    assert second["from_cache"] is True
    assert second["top"] == first["top"]
    assert fake_predictor.calls == 1  # the model ran only once


def test_different_images_are_both_misses(client: FlaskClient) -> None:
    upload(client, make_image_bytes(color=(10, 200, 10)))
    second = upload(client, make_image_bytes(color=(200, 10, 10))).get_json()
    assert second["from_cache"] is False


def test_low_confidence_returns_not_sure() -> None:
    from app import create_app

    unsure = FakePredictor(probabilities=confident_probabilities(best=0.3))
    client = create_app(LeafCheckService(unsure)).test_client()
    body = upload(client, make_image_bytes()).get_json()
    assert body["status"] == "not_sure"
    assert body["prediction"] is None
    assert body["message"] == config.NOT_SURE_MESSAGE


def test_stats_counts_update(client: FlaskClient) -> None:
    image = make_image_bytes()
    upload(client, image)
    upload(client, image)
    upload(client, make_image_bytes(color=(1, 2, 3)))
    stats = client.get("/stats?format=json").get_json()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["cache_size"] == 2
    assert stats["cache_capacity"] == 4
    assert stats["hit_rate"] == round(1 / 3, 4)


def test_failed_uploads_do_not_count_in_stats(client: FlaskClient) -> None:
    upload(client, b"")
    assert client.get("/stats?format=json").get_json()["requests"] == 0


def test_stats_page_renders(client: FlaskClient) -> None:
    upload(client, make_image_bytes())
    response = client.get("/stats")
    assert response.status_code == 200
    assert "Hit rate" in response.get_data(as_text=True)


def test_health(client: FlaskClient) -> None:
    body = client.get("/health").get_json()
    assert body == {"status": "ok", "demo_mode": True, "classes": 19}


def test_result_includes_leaf_analysis_and_heatmap_field(client: FlaskClient) -> None:
    body = upload(client, make_image_bytes()).get_json()
    assert set(body["leaf_analysis"]) >= {"affected_percent", "leaf_percent", "warnings", "severity_band"}
    assert body["heatmap"] is None  # the fake predictor has no heatmap


def test_gallery_sample_route_only_serves_listed_files(client: FlaskClient) -> None:
    assert client.get("/samples/tomato_healthy.jpg").status_code == 200
    assert client.get("/samples/CREDITS.md").status_code == 404
    assert client.get("/samples/..%2Fapp.py").status_code == 404
