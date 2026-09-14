from train_model import data, cat_features, y_test, preds, model, X_test
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import pytest


def test_process_data_shape():
    """
    Tests if original data has the same number of rows as the processed data.

    """
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    assert X.shape[0] == data.shape[0]
    assert y.shape[0] == data.shape[0]


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # Tests if the model metrics are all float values and are the expected values.
    """
    p, r, fb = compute_model_metrics(y_test, preds)
    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(fb, float)
    assert p == pytest.approx(0.7616, abs=0.0001)
    assert r == pytest.approx(0.2094, abs=0.0001)
    assert fb == pytest.approx(0.3285, abs=0.0001)


def test_prediction():
    """
    Tests that prediction function returns 0 or 1.
    """
    preds = inference(model, X_test)
    assert all(pred in [0, 1] for pred in preds)
