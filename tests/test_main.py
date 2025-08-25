# test_main.py
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def import_app(monkeypatch):
    """
    Import app.main while stubbing predict_one and telemetry client.
    """
    import sys
    from pathlib import Path

    # Make sure 'app' package is importable
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import importlib
    import app.main as main

    # Stub predict_one (avoid touching real TF)
    def fake_predict_one(text: str):
        return {
            "sentiment": "pos",
            "proba_neg": 0.1,
            "proba_pos": 0.9,
            "model_version": "test:v0",
        }

    main.predict_one = fake_predict_one

    # Stub telemetry client with a tiny spy object
    class FakeTC:
        def __init__(self):
            self.events = []

        def track_event(self, name, props=None):
            self.events.append((name, props or {}))

        def flush(self):
            pass

    main.telemetry_client = FakeTC()
    return main


@pytest.fixture
def client(import_app):
    from fastapi.testclient import TestClient
    return TestClient(import_app.app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_endpoint(client, import_app):
    payload = {"text": "I love it"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["sentiment"] == "pos"
    assert 0.0 <= body["proba_pos"] <= 1.0
    assert "model_version" in body

    # Telemetry recorded
    tc = import_app.telemetry_client
    assert any(evt[0] == "prediction" for evt in tc.events)


def test_feedback_ok(client, import_app):
    payload = {
        "text": "Nice",
        "predicted": "pos",
        "correct": True,
        "note": "all good"
    }
    res = client.post("/feedback", json=payload)
    assert res.status_code == 200
    assert res.json() == {"status": "stored"}

    tc = import_app.telemetry_client
    assert any(evt[0] == "feedback_ok" for evt in tc.events)


def test_feedback_bad_prediction(client, import_app):
    payload = {
        "text": "Nope",
        "predicted": "pos",
        "correct": False,
        "note": "misclassified"
    }
    res = client.post("/feedback", json=payload)
    assert res.status_code == 200
    assert res.json() == {"status": "stored"}

    tc = import_app.telemetry_client
    assert any(evt[0] == "bad_prediction" for evt in tc.events)


def test_home_404_when_ui_missing(import_app, client):
    # Ensure static/index.html does not exist
    index = import_app.INDEX
    if index.exists():
        index.unlink()
    if index.parent.exists():
        # keep folder or not; not required
        pass

    res = client.get("/")
    assert res.status_code == 404
    detail = res.json()["detail"]
    # Message must include the absolute path in quotes, as in main.py
    assert str(import_app.INDEX) in detail


def test_home_serves_ui(import_app, client):
    # Create static/index.html next to app
    index: Path = import_app.INDEX
    index.parent.mkdir(exist_ok=True)
    index.write_text("<!doctype html><title>OK</title>", encoding="utf-8")

    try:
        res = client.get("/")
        assert res.status_code == 200
        # FileResponse => HTML body returned
        assert "OK" in res.text
    finally:
        # Cleanup
        if index.exists():
            index.unlink()
