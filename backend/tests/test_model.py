import os
import mlflow.pyfunc

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model_local")


def test_loaded_model():
    loaded_model = mlflow.pyfunc.load_model(MODEL_DIR)
    input_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]
    predictions = loaded_model.predict(input_data).tolist()
    assert isinstance(predictions, list)
    assert len(predictions) == len(input_data)
    assert all(isinstance(x, (int, float)) for x in predictions)
