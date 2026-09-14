from train_model import data, cat_features, model, X_test
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


def test_compute_model_metrics():
    """
    # Tests compute_model_metrics with known labels and predictions
    """
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0,]
    p, r, fb = compute_model_metrics(y_true, y_pred)
    assert p == pytest.approx(2 / 3, abs=0.0001)
    assert r == pytest.approx(2 / 3, abs=0.0001)
    assert fb == pytest.approx(2 / 3, abs=0.0001)


def test_prediction():
    """
    Tests that prediction function returns 0 or 1.
    """
    preds = inference(model, X_test)
    assert all(pred in [0, 1] for pred in preds)
