"""The only test that loads the real model. Skipped if the model is missing.

It runs the model on the held-out test images exported by
notebooks/evaluate.ipynb into samples/test_split/ (with expected.json mapping
file name -> true class) and checks accuracy is above 85%.
"""

import json

import pytest

from leafcheck import config
from leafcheck.preprocess import to_model_input

TEST_SPLIT_DIR = config.SAMPLES_DIR / "test_split"
EXPECTED_PATH = TEST_SPLIT_DIR / "expected.json"

pytestmark = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.LABELS_PATH.exists() and EXPECTED_PATH.exists()),
    reason="trained model or samples/test_split not present",
)


def test_model_accuracy_on_sample_test_images() -> None:
    pytest.importorskip("tensorflow")
    from leafcheck.predictor import KerasPredictor

    predictor = KerasPredictor()
    expected: dict[str, str] = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    correct = 0
    for filename, true_label in expected.items():
        probabilities = predictor.predict_proba(to_model_input((TEST_SPLIT_DIR / filename).read_bytes()))
        correct += predictor.labels[int(probabilities.argmax())] == true_label
    accuracy = correct / len(expected)
    assert accuracy > 0.85, f"accuracy {accuracy:.2%} on {len(expected)} images"
