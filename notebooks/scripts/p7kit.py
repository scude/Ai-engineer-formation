# scripts/p7kit.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Dict, Tuple
import os, logging, json
import numpy as np
import pandas as pd
import mlflow
import mlflow.data

# ---------------- Paths & logging ----------------
def get_paths(nb_dir: Path | None = None) -> Tuple[Path, Path, Path, Path]:
    nb_dir = nb_dir or Path(".").resolve()          # notebooks/
    base   = nb_dir.parent                          # projet7/
    data   = base / "data"
    mlruns = base / "mlruns"
    emb    = base / "embeddings"; emb.mkdir(parents=True, exist_ok=True)
    return nb_dir, data, mlruns, emb

def silent_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

def silent_transformers():
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    try:
        from datasets.utils.logging import disable_progress_bar, set_verbosity
        disable_progress_bar(); set_verbosity(logging.ERROR)
    except Exception:
        pass

# ---------------- Data I/O ----------------
def pick_or_csv(maybe_df: Optional[pd.DataFrame], csv_path: Path) -> pd.DataFrame:
    return maybe_df.copy() if isinstance(maybe_df, pd.DataFrame) else pd.read_csv(csv_path)

def ensure_cols(df: pd.DataFrame, text_col="text", label_col="label") -> pd.DataFrame:
    return df[[text_col, label_col]].rename(columns={text_col: "text"}).dropna().copy()

def ensure_int_labels(df: pd.DataFrame, label_col="label") -> pd.DataFrame:
    out = df.copy(); out[label_col] = out[label_col].astype(int); return out

# ---------------- MLflow (unifiés) ----------------
def mlflow_setup(mlruns_dir: Path, experiment: str) -> None:
    mlflow.set_tracking_uri(f"file://{mlruns_dir.resolve()}")
    mlflow.set_experiment(experiment)

class run:
    """Context manager léger: with p7.run('name'): ..."""
    def __init__(self, run_name: Optional[str] = None): self.run_name = run_name
    def __enter__(self): return mlflow.start_run(run_name=self.run_name)
    def __exit__(self, exc_type, exc, tb): mlflow.end_run()

def log_params(d: Dict) -> None:
    mlflow.log_params({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in d.items()})

def log_metrics(d: Dict[str, float]) -> None:
    mlflow.log_metrics({k: float(v) for k, v in d.items()})

def make_mlflow_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    int_cols = df.select_dtypes(include=["int", "int32", "int64"]).columns
    if len(int_cols) > 0: df[int_cols] = df[int_cols].astype("float64")
    return df

def log_inputs(train_df: pd.DataFrame | None = None,
               val_df:   pd.DataFrame | None = None,
               test_df:  pd.DataFrame | None = None) -> None:
    if train_df is not None: mlflow.log_input(mlflow.data.from_pandas(train_df), context="train_raw")
    if val_df   is not None: mlflow.log_input(mlflow.data.from_pandas(val_df),   context="val_raw")
    if test_df  is not None: mlflow.log_input(mlflow.data.from_pandas(test_df),  context="test_raw")

def log_artifacts_dir(path: Path, artifact_path: str) -> None:
    mlflow.log_artifacts(str(path), artifact_path=artifact_path)

# ---------------- Reporting & artefacts ----------------
def save_classification_artifacts(y_true: Iterable[int], y_pred: Iterable[int], out_dir: Path,
                                  labels=("neg", "pos")) -> None:
    from sklearn.metrics import classification_report, confusion_matrix
    from itertools import product
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion matrix"); plt.colorbar()
    plt.xticks([0,1], labels); plt.yticks([0,1], labels)
    for i, j in product(range(2), range(2)): plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png"); plt.close(fig)

def prepare_out_dir(base: Path, run_name: str) -> Path:
    p = base / run_name; p.mkdir(parents=True, exist_ok=True); return p

def fmt_metrics_line(d: Dict[str, float]) -> str:
    parts = [f"{k}={v:.4f}" for k, v in d.items()]
    return " | ".join(parts)

def banner_ok(tag: str, out_dir: Path, metrics_show: Dict[str, float]) -> None:
    line = fmt_metrics_line(metrics_show) if metrics_show else ""
    print(f"✅ {tag} — OK" + (f" | {line}" if line else ""))
    print(f"   Saved to: {out_dir.resolve()}")

# ---------------- tf.data (générique) ----------------
def make_tfds(X, y, batch_size: int, seed: int, training: bool = False):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training: ds = ds.shuffle(min(len(X), 50_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
