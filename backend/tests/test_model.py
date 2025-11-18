import mlflow.pyfunc
import pandas as pd
from sklearn import datasets

MODEL_DIR = "model"


def test_loaded_model():
    loaded_model = mlflow.pyfunc.load_model(MODEL_DIR)

    X, y = datasets.load_iris(return_X_y=True)
    feature_names = datasets.load_iris().feature_names

    df = pd.DataFrame(X, columns=feature_names)

    preds = loaded_model.predict(df)
    result = df.copy()
    result["actual"] = y
    result["predicted"] = preds

    print(result.head())
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_loaded_model()
