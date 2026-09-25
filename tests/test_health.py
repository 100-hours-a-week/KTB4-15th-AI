from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_health_is_public_liveness_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "CHECKPOINT_DSN", "")
    monkeypatch.setattr(settings, "AUTH_DISABLED", False)
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "required-for-other-routes")
    monkeypatch.setenv("RUNWARE_VTON_API_KEY", "test-vton-key")
    monkeypatch.setenv("RUNWARE_LLM_API_KEY", "test-llm-key")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
