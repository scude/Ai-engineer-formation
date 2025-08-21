from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

ART = Path(__file__).parent / "artifacts"
MODEL_PATH = ART / "model.keras"
TOK_PATH   = ART / "tokenizer.json"

# Must match training
MAX_LEN = int(os.getenv("MAX_LEN", 80))

# --- Load at startup (singletons) ---
with TOK_PATH.open("r", encoding="utf-8") as f:
    tok = tokenizer_from_json(f.read())

model = keras.models.load_model(MODEL_PATH)

MODEL_VERSION = os.getenv("MODEL_VERSION", "keras_cnn_bilstm:v1")

def predict_one(text: str) -> Dict[str, Any]:
    """
    Run binary sentiment inference with Keras CNN+BiLSTM model.
    Returns sentiment label and probabilities for neg/pos.
    """
    # 1) text -> seq ids using the SAME tokenizer as training
    seq = tok.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # 2) model -> sigmoid proba for positive class
    p_pos = float(model.predict(x, verbose=0).ravel()[0])
    p_neg = 1.0 - p_pos
    sentiment = "pos" if p_pos >= 0.5 else "neg"

    return {
        "sentiment": sentiment,
        "proba_neg": p_neg,
        "proba_pos": p_pos,
        "model_version": MODEL_VERSION,
    }
