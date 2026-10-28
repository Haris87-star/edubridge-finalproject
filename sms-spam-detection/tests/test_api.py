"""API tests: run against an isolated app instance with a synthetic model.

No network, no real artifact, no shared state between tests.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.metrics import MetricsRegistry
from api.service import ModelService
from api.settings import Settings
from src.model import save_model


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        model_path=tmp_path / "model.joblib",
        threshold=0.5,
        max_batch_size=5,
        admin_token=None,
        feedback_path=tmp_path / "feedback" / "feedback.jsonl",
        autotrain=False,
    )


@pytest.fixture()
def service(settings, trained_classifier) -> ModelService:
    return ModelService(settings, classifier=trained_classifier)


@pytest.fixture()
def client(settings, service) -> TestClient:
    app = create_app(settings=settings, service=service, metrics=MetricsRegistry())
    with TestClient(app) as test_client:
        yield test_client


SPAM_TEXT = "congratulations you won cash prize call now"
HAM_TEXT = "see you at lunch tomorrow"


# ------------------------------------------------------------------ ops routes
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 0
    assert body["version"]


def test_ready_reports_model(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert len(body["version"]) >= 8


def test_unknown_route_returns_404(client):
    assert client.get("/nope").status_code == 404


def test_web_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SMS Spam Detector" in response.text


def test_openapi_schema_is_available(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]


# -------------------------------------------------------------------- predict
def test_predict_spam(client):
    response = client.post("/predict", json={"text": SPAM_TEXT})
    assert response.status_code == 200

    body = response.json()
    assert body["label"] == "spam"
    assert body["spam_probability"] > 0.5
    assert body["confidence"] >= 0.5
    assert body["threshold"] == 0.5
    assert body["model"]["version"] != "unknown"
    assert body["request_id"] == response.headers["x-request-id"]
    assert float(response.headers["x-response-time-ms"]) >= 0


def test_predict_ham(client):
    body = client.post("/predict", json={"text": HAM_TEXT}).json()
    assert body["label"] == "ham"
    assert body["spam_probability"] < 0.5


def test_predict_explanations(client):
    body = client.post("/predict", json={"text": SPAM_TEXT, "explain": True}).json()
    explanation = body["explanation"]

    assert explanation, "expected token contributions"
    assert all({"token", "weight"} <= set(item) for item in explanation)
    assert explanation[0]["weight"] > 0
    assert len(explanation) <= 8


def test_threshold_override(client):
    # A threshold of 0 makes every message spam; probability >= 0 always holds.
    body = client.post("/predict", json={"text": HAM_TEXT, "threshold": 0.0}).json()
    assert body["label"] == "spam"
    assert body["threshold"] == 0.0


def test_blank_text_is_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_missing_field_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def test_threshold_out_of_range_is_rejected(client):
    assert client.post("/predict", json={"text": SPAM_TEXT, "threshold": 1.5}).status_code == 422


def test_custom_request_id_is_echoed(client):
    response = client.post(
        "/predict", json={"text": HAM_TEXT}, headers={"X-Request-ID": "test-123"}
    )
    assert response.json()["request_id"] == "test-123"


# ---------------------------------------------------------------------- batch
def test_batch_prediction(client):
    messages = [SPAM_TEXT, HAM_TEXT, "free cash prize win now"]
    response = client.post("/predict/batch", json={"messages": messages})
    assert response.status_code == 200

    body = response.json()
    assert len(body["results"]) == 3
    assert [item["message"] for item in body["results"]] == messages
    assert body["summary"]["total"] == 3
    assert body["summary"]["spam"] + body["summary"]["ham"] == 3
    assert body["summary"]["spam"] >= 1
    assert body["model"]["name"]
    assert body["latency_ms"] >= 0


def test_batch_size_limit(client):
    response = client.post("/predict/batch", json={"messages": ["hi"] * 6})
    assert response.status_code == 413
    assert "maximum is 5" in response.json()["detail"]


def test_batch_rejects_blank_entries(client):
    response = client.post("/predict/batch", json={"messages": ["ok", "  "]})
    assert response.status_code == 422


def test_batch_explanations_optional(client):
    body = client.post(
        "/predict/batch", json={"messages": [SPAM_TEXT], "explain": True}
    ).json()
    assert body["results"][0]["explanation"]


# ----------------------------------------------------------------- model info
def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["name"]
    assert body["version"]
    assert body["threshold"] == 0.5
    assert "accuracy" in body["metrics"]
    assert body["leaderboard"]
    assert body["dataset"]["rows"] > 0


# --------------------------------------------------------------------- mlops
def test_metrics_endpoint(client):
    client.post("/predict", json={"text": SPAM_TEXT})
    client.post("/predict", json={"text": HAM_TEXT})

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    text = response.text
    assert 'spam_api_predictions_total{label="spam"}' in text
    assert 'spam_api_predictions_total{label="ham"}' in text
    assert "spam_api_requests_total{" in text
    assert "spam_api_uptime_seconds" in text


def test_stats_endpoint(client):
    client.get("/health")
    body = client.get("/stats").json()
    assert body["requests_total"] >= 1
    assert "predictions_total" in body


def test_feedback_is_persisted(client, settings):
    response = client.post(
        "/feedback",
        json={"text": SPAM_TEXT, "actual": "spam", "predicted": "ham", "notes": "missed"},
    )
    assert response.status_code == 201
    assert response.json()["stored"] is True

    lines = settings.feedback_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["actual"] == "spam"
    assert record["predicted"] == "ham"
    assert "received_at" in record


def test_feedback_validates_labels(client):
    response = client.post("/feedback", json={"text": "hello", "actual": "maybe"})
    assert response.status_code == 422


def test_admin_reload_disabled_without_token(client):
    assert client.post("/admin/reload").status_code == 403


def test_admin_reload_requires_token(settings, trained_classifier):
    secure = dataclasses.replace(settings, admin_token="s3cret")
    save_model(trained_classifier, secure.model_path)

    app = create_app(
        settings=secure,
        service=ModelService(secure, classifier=trained_classifier),
        metrics=MetricsRegistry(),
    )
    with TestClient(app) as client:
        assert client.post("/admin/reload").status_code == 401
        assert (
            client.post("/admin/reload", headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )

        response = client.post(
            "/admin/reload", headers={"Authorization": "Bearer s3cret"}
        )
        assert response.status_code == 200
        assert response.json() == {
            "reloaded": True,
            "version": trained_classifier.version,
        }


def test_admin_reload_missing_artifact_returns_404(settings, trained_classifier):
    secure = dataclasses.replace(settings, admin_token="s3cret")
    app = create_app(
        settings=secure,
        service=ModelService(secure, classifier=trained_classifier),
        metrics=MetricsRegistry(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/admin/reload", headers={"Authorization": "Bearer s3cret"}
        )
    assert response.status_code == 404


# ------------------------------------------------------------- not-ready state
def test_predict_without_model_returns_503(tmp_path):
    settings = Settings(
        model_path=tmp_path / "absent.joblib",
        autotrain=False,
        feedback_path=tmp_path / "feedback.jsonl",
    )
    app = create_app(settings=settings, service=ModelService(settings), metrics=MetricsRegistry())

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", json={"text": "hello"})
        assert response.status_code == 503
        assert "python main.py train" in response.json()["detail"]
