from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_health_check():
    """Verify that root metadata health endpoint returns healthy status code."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "endpoints" in data or "ensemble_metadata" in data


def test_predict_single_schema_validation():
    """Verify that malformed payloads are rejected with 422 Unprocessable Entity."""
    # Missing mandatory Driver name field
    malformed_payload = {
        "Race": "Monaco",
        "Compound": "SOFT",
        "LapNumber": 15,
        "Stint": 1,
        "TyreLife": 12.0,
        "Position": 3.0,
        "LapTime (s)": 78.5,
        "LapTime_Delta": 0.2,
        "Cumulative_Degradation": 0.08,
        "RaceProgress": 0.25
    }
    response = client.post("/predict/single", json=malformed_payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_single_numeric_bounds():
    """Verify that values violating Pydantic numerical constraints are caught."""
    # LapNumber is ge=1, setting it to 0 is invalid
    boundary_payload = {
        "Driver": "Hamilton",
        "Race": "Monaco",
        "Compound": "SOFT",
        "LapNumber": 0,  # Invalid
        "Stint": 1,
        "TyreLife": 12.0,
        "Position": 3.0,
        "LapTime (s)": 78.5,
        "LapTime_Delta": 0.2,
        "Cumulative_Degradation": 0.08,
        "RaceProgress": 0.25,
        "Position_Change": 0.0
    }
    response = client.post("/predict/single", json=boundary_payload)
    assert response.status_code == 422
