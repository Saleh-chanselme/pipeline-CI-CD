from fastapi.testclient import TestClient
from app.ml_api import app

client = TestClient(app)


def test_predict_endpoint():
    """
    Test the /predict endpoint with a sample input.
    """
    data = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data, "Response JSON missing 'predictions' key"
    predictions = json_data["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert all(isinstance(x, (int, float)) for x in predictions)
