"""Misinformation / fake-news classifier on LIAR (step 6).

Usage (project root, venv active):
    python -m src.misinfo                 # trains baselines, prints test metrics, saves model + results
Trains TF-IDF + Logistic Regression for (a) binary fake/real and (b) the original 6 labels.
Hyper-parameter C is picked on the validation split; final numbers are reported on the test split.
"""
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline

from .config import PROCESSED, ROOT, SEED
from .load_data import load_liar

ORDER = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]


def make_pipe(C: float):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def fit_task(target: str, tr, va, te) -> dict:
    best_c, best_f1 = None, -1.0
    for C in (0.3, 1, 3, 10):
        p = make_pipe(C).fit(tr.statement, tr[target])
        f = f1_score(va[target], p.predict(va.statement), average="macro")
        if f > best_f1:
            best_c, best_f1 = C, f
    pipe = make_pipe(best_c).fit(pd.concat([tr.statement, va.statement]), pd.concat([tr[target], va[target]]))
    pred = pipe.predict(te.statement)
    print(f"\n=== {target}  (C={best_c}) ===")
    print(classification_report(te[target], pred, digits=3, zero_division=0))
    return {"task": target, "C": best_c, "n_test": len(te),
            "accuracy": round(accuracy_score(te[target], pred), 4),
            "macro_f1": round(f1_score(te[target], pred, average="macro"), 4)}, pipe


def main():
    tr, va, te = (load_liar(s, binary=True) for s in ("train", "valid", "test"))
    results, PROCESSED_MODELS = [], ROOT / "models"
    PROCESSED_MODELS.mkdir(exist_ok=True)
    for target in ("label_bin", "label"):
        res, pipe = fit_task(target, tr, va, te)
        results.append(res)
        joblib.dump(pipe, PROCESSED_MODELS / f"liar_{target}.joblib")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(PROCESSED / "benchmark_liar.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
