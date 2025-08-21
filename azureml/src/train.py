from __future__ import annotations
import os, argparse, numpy as np, pandas as pd, mlflow
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_weighted": f1_score(labels, preds, average="weighted")}

def load_split_csv(path: str, text_col: str, label_col: str, sample_n: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna().copy()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    df["label"] = df["label"].astype(int)
    if sample_n and sample_n > 0:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
    return df.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  default="")
    ap.add_argument("--text_col",  default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--epochs",    type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--sample_n",   type=int, default=0)  # 0 = full
    args = ap.parse_args()

    if uri := os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("p7-distilbert")

    with mlflow.start_run(run_name="distilbert-cpu-splits"):
        ckpt = "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(ckpt)

        tr_df = load_split_csv(args.train_csv, args.text_col, args.label_col, args.sample_n)
        va_df = load_split_csv(args.val_csv,   args.text_col, args.label_col, args.sample_n)
        te_df = load_split_csv(args.test_csv,  args.text_col, args.label_col, args.sample_n) if args.test_csv else None

        ds_train = Dataset.from_pandas(tr_df, preserve_index=False)
        ds_val   = Dataset.from_pandas(va_df, preserve_index=False)
        ds_test  = Dataset.from_pandas(te_df, preserve_index=False) if te_df is not None else None

        def tok_fn(b): return tok(b["text"], truncation=True, max_length=args.max_length)
        ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
        ds_val   = ds_val.map(tok_fn,   batched=True, remove_columns=["text"])
        if ds_test is not None:
            ds_test = ds_test.map(tok_fn, batched=True, remove_columns=["text"])

        collator = DataCollatorWithPadding(tokenizer=tok)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

        targs = TrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=args.epochs,
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_strategy="no",
            report_to=[]
        )

        trainer = Trainer(model=model, args=targs,
                          train_dataset=ds_train, eval_dataset=ds_val,
                          tokenizer=tok, data_collator=collator,
                          compute_metrics=compute_metrics)
        trainer.train()
        val_metrics = trainer.evaluate()
        mlflow.log_metrics({f"val_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))})

        if ds_test is not None:
            preds = trainer.predict(ds_test)
            y_true = preds.label_ids
            y_pred = np.argmax(preds.predictions, axis=-1)
            mlflow.log_metrics({
                "test_accuracy": accuracy_score(y_true, y_pred),
                "test_f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            })

        out_dir = os.environ.get("AZUREML_OUTPUT_MODEL_DIR", "./outputs/model")
        os.makedirs(out_dir, exist_ok=True)
        trainer.save_model(out_dir); tok.save_pretrained(out_dir)
        mlflow.log_artifacts(out_dir, artifact_path="model")

if __name__ == "__main__":
    os.makedirs("./outputs", exist_ok=True)
    main()
