"""Shared pytest fixtures: a fake predictor and a Flask test client.

No test here needs TensorFlow or the trained model.
"""

from typing import Iterator

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app
from leafcheck import config
from leafcheck.predictor import FakePredictor
from leafcheck.service import LeafCheckService


def confident_probabilities(best_index: int = 10, best: float = 0.9) -> list[float]:
    """19 probabilities with one clear winner (default: Tomato early blight)."""
    n = len(config.CLASS_NAMES)
    values = [(1.0 - best) / (n - 1)] * n
    values[best_index] = best
    return values


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor(probabilities=confident_probabilities())


@pytest.fixture
def service(fake_predictor: FakePredictor) -> LeafCheckService:
    return LeafCheckService(fake_predictor, demo_mode=True, cache_capacity=4)


@pytest.fixture
def app(service: LeafCheckService) -> Flask:
    flask_app = create_app(service)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app: Flask) -> Iterator[FlaskClient]:
    with app.test_client() as test_client:
        yield test_client
