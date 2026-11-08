from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_reports_model_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_triage_predicts_a_category():
    response = client.post(
        "/triage",
        json={"text": "I was charged twice for my monthly subscription"},
    )
    assert response.status_code == 200
    assert response.json()["category"] == "billing"
    assert 0 <= response.json()["confidence"] <= 1


def test_triage_rejects_short_text():
    response = client.post("/triage", json={"text": "help"})
    assert response.status_code == 422
