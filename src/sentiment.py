"""Transformer sentiment analysis (step 3).

Usage (from project root, venv active):
    python -m src.sentiment --dataset reddit_comments --n 5000
    python -m src.sentiment --dataset reddit_posts
    python -m src.sentiment --benchmark --n 5000        # Sentiment140 accuracy/F1
Models: 'roberta' (3-class, tweet-trained, best quality) or 'fast' (DistilBERT, binary, ~2-3x faster on CPU).
"""
import argparse
import re

import numpy as np
import pandas as pd

from .config import PROCESSED, SEED
from .load_data import load_reddit_comments, load_reddit_posts, load_sentiment140

MODELS = {
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "fast": "distilbert-base-uncased-finetuned-sst-2-english",
}
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")


def preprocess(text) -> str:
    """Normalise like the tweet-trained model expects: @user and http placeholders."""
    if not isinstance(text, str):
        return ""
    t = _MENTION.sub("@user", text)
    t = _URL.sub("http", t)
    return re.sub(r"\s+", " ", t).strip()


def predict(texts, model_key: str = "roberta", batch_size: int = 32, max_length: int = 128) -> pd.DataFrame:
    """Return DataFrame with p_negative, p_neutral, p_positive, label, sent_score (= p_pos - p_neg)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    id2label = {int(i): l.lower() for i, l in model.config.id2label.items()}

    texts = [preprocess(t) or "." for t in texts]
    order = np.argsort([len(t) for t in texts])          # length-sorted batches = much faster
    probs = np.zeros((len(texts), 3), dtype=np.float32)  # columns: negative, neutral, positive
    col = {"negative": 0, "neutral": 1, "positive": 2}

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="sentiment"):
            idx = order[start:start + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            p = torch.softmax(model(**enc).logits, dim=-1).numpy()
            for j, lab in id2label.items():
                probs[idx, col[lab]] = p[:, j]

    out = pd.DataFrame(probs, columns=["p_negative", "p_neutral", "p_positive"])
    out["label"] = out[["p_negative", "p_neutral", "p_positive"]].idxmax(axis=1).str[2:]
    out["sent_score"] = out.p_positive - out.p_negative
    return out


def run_dataset(name: str, n: int | None, model_key: str, batch_size: int):
    df = load_reddit_comments() if name == "reddit_comments" else load_reddit_posts()
    df = df[df.text.str.len() > 0]
    if n:
        df = df.sample(min(n, len(df)), random_state=SEED)
    df = df.reset_index(drop=True)
    res = predict(df.text.tolist(), model_key, batch_size)
    out = pd.concat([df, res], axis=1)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}_sentiment.parquet"
    out.to_parquet(path)
    print(f"saved {len(out)} rows -> {path}")
    return out


def benchmark_sentiment140(n_per_class: int, model_key: str, batch_size: int) -> dict:
    """Binary accuracy/F1 on a balanced Sentiment140 sample (the file is sorted by label, so we sample)."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    df = load_sentiment140()
    df = df.groupby("label", group_keys=False).sample(n_per_class, random_state=SEED).reset_index(drop=True)
    res = predict(df.text.tolist(), model_key, batch_size)
    # Sentiment140 has no neutral class: decide by the larger of p_pos / p_neg
    pred = np.where(res.p_positive >= res.p_negative, "positive", "negative")
    acc = accuracy_score(df.label, pred)
    f1 = f1_score(df.label, pred, pos_label="positive")
    print(classification_report(df.label, pred, digits=3))
    out = {"model": MODELS[model_key], "n": len(df), "accuracy": round(acc, 4), "f1_positive": round(f1, 4)}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(PROCESSED / f"benchmark_sentiment140_{model_key}.csv", index=False)
    print(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["reddit_comments", "reddit_posts"])
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--model", choices=list(MODELS), default="roberta")
    ap.add_argument("--n", type=int, default=None, help="rows (or rows per class for --benchmark)")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    if a.benchmark:
        benchmark_sentiment140(a.n or 5000, a.model, a.batch_size)
    elif a.dataset:
        run_dataset(a.dataset, a.n, a.model, a.batch_size)
    else:
        ap.error("choose --dataset or --benchmark")
