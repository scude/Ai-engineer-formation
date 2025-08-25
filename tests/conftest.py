# conftest.py
import json
import importlib
from pathlib import Path
import pytest
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    # Keep logs quiet and ensure deterministic max length
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "2")
    monkeypatch.setenv("MAX_LEN", "80")
    # No real AppInsights key during tests
    monkeypatch.delenv("APPINSIGHTS_INSTRUMENTATIONKEY", raising=False)
    yield


@pytest.fixture
def project_root() -> Path:
    # Points to the folder that contains app/*.py
    # Adjust if your package name differs
    return Path(__file__).resolve().parents[1] / "app"


@pytest.fixture
def ensure_artifacts(project_root: Path):
    """
    Create a minimal-but-valid Keras Tokenizer JSON so that the real
    tokenizer_from_json() can parse it without error.
    """
    art = project_root / "artifacts"
    art.mkdir(exist_ok=True)

    tok_path = art / "tokenizer.json"
    # Minimal valid structure for Keras legacy tokenizer_from_json()
    tokenizer_json = {
        "class_name": "Tokenizer",
        "config": {
            "num_words": None,
            "filters": '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            "lower": True,
            "split": " ",
            "char_level": False,
            "oov_token": None,
            "document_count": 1,
            # These 5 fields must be JSON-encoded strings
            "word_counts": json.dumps({"hello": 1}),
            "word_docs": json.dumps({"hello": 1}),
            "index_docs": json.dumps({}),
            "word_index": json.dumps({"hello": 1}),
            "index_word": json.dumps({"1": "hello"}),
        },
    }
    tok_path.write_text(json.dumps(tokenizer_json), encoding="utf-8")
    # On ne crée PAS de vrai model.keras : on mockera load_model
    yield



@pytest.fixture
def fake_tokenizer_class():
    class FakeTokenizer:
        def texts_to_sequences(self, texts):
            # Very simple deterministic mapping ignoring input
            return [[1, 2, 3]]
    return FakeTokenizer


@pytest.fixture
def fake_model_class():
    import numpy as np
    class FakeModel:
        def __init__(self, value=0.8):
            self.value = value
        def predict(self, x, verbose=0):
            # IMPORTANT: return a NumPy array, not a list
            return np.array([[self.value]], dtype=float)
    return FakeModel


@pytest.fixture
def import_inference_with_mocks(monkeypatch, project_root, ensure_artifacts, fake_model_class):
    import sys, importlib
    monkeypatch.setattr(
        "tensorflow.keras.models.load_model",
        lambda *_args, **_kwargs: fake_model_class(),
        raising=True,
    )
    root = project_root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import app.inference as inference
    inference = importlib.reload(inference)
    return inference
