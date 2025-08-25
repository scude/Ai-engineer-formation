# test_inference.py
import importlib
import numpy as np


def test_predict_one_positive(import_inference_with_mocks, fake_model_class):
    inference = import_inference_with_mocks

    # Ensure our default fake model returns 0.8 -> "pos"
    res = inference.predict_one("Great product!")
    assert res["sentiment"] == "pos"
    assert 0.0 <= res["proba_pos"] <= 1.0
    assert np.isclose(res["proba_pos"], 0.8, atol=1e-6)
    assert np.isclose(res["proba_neg"], 0.2, atol=1e-6)
    assert "model_version" in res


def test_predict_one_negative(import_inference_with_mocks, fake_model_class, monkeypatch):
    inference = import_inference_with_mocks

    # Swap the global model for a negative case (0.2 -> "neg")
    inference.model = fake_model_class(value=0.2)

    res = inference.predict_one("Terrible experience.")
    assert res["sentiment"] == "neg"
    assert np.isclose(res["proba_pos"], 0.2, atol=1e-6)
    assert np.isclose(res["proba_neg"], 0.8, atol=1e-6)
