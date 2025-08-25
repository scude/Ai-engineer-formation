# test_schemas.py
import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest, FeedbackRequest, PredictResponse


def test_predict_request_min_length():
    with pytest.raises(ValidationError):
        PredictRequest(text="")  # min_length=1


def test_predict_request_ok():
    req = PredictRequest(text="Hello")
    assert req.text == "Hello"


def test_feedback_request_required_fields():
    fb = FeedbackRequest(text="abc", predicted="pos", correct=False, note=None)
    assert fb.predicted in {"pos", "neg"}
    assert fb.correct is False


def test_predict_response_shape():
    # Just ensure fields exist, values are typed correctly
    pr = PredictResponse(
        sentiment="pos",
        proba_neg=0.2,
        proba_pos=0.8,
        model_version="x:v1",
    )
    assert pr.sentiment in {"pos", "neg"}
    assert 0.0 <= pr.proba_pos <= 1.0
    assert 0.0 <= pr.proba_neg <= 1.0