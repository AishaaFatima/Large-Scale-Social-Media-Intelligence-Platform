"""Topic discovery with BERTopic + topics over time (step 4).

Usage (project root, venv active):
    python -m src.topics --n 3000      # quick test
    python -m src.topics               # all Reddit posts (~20k)
Outputs in data/processed/: topic_info.csv, post_topics.parquet, topics_over_time.csv
and an interactive chart in docs/topics_over_time.html
"""
import argparse
import re

import pandas as pd

from .config import PROCESSED, ROOT, SEED
from .load_data import load_reddit_posts

_URL = re.compile(r"https?://\S+|www\.\S+")


def prepare_docs(df: pd.DataFrame, max_chars: int = 1000) -> pd.DataFrame:
    df = df.copy()
    df["doc"] = df["text"].fillna("").map(lambda t: _URL.sub(" ", t)[:max_chars].strip())
    df["doc"] = df["doc"].str.replace(r"\[(deleted|removed)\]", " ", regex=True).str.strip()
    return df[df["doc"].str.len() >= 15].reset_index(drop=True)


def run(n: int | None = None, min_topic_size: int = 15, nr_bins: int = 24):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    df = prepare_docs(load_reddit_posts())
    if n:
        df = df.sample(min(n, len(df)), random_state=SEED).reset_index(drop=True)
    docs, times = df["doc"].tolist(), df["created"].tolist()
    print(f"{len(docs)} documents")

    model = BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
        min_topic_size=min_topic_size,
        verbose=True,
    )
    topics, _ = model.fit_transform(docs)
    df["topic"] = topics

    info = model.get_topic_info()
    tot = model.topics_over_time(docs, times, nr_bins=nr_bins)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)
    info.to_csv(PROCESSED / "topic_info.csv", index=False)
    df.drop(columns=["doc"]).to_parquet(PROCESSED / "post_topics.parquet")
    tot.to_csv(PROCESSED / "topics_over_time.csv", index=False)
    model.visualize_topics_over_time(tot, top_n_topics=10).write_html(ROOT / "docs" / "topics_over_time.html")
    print(info[["Topic", "Count", "Name"]].head(15).to_string(index=False))
    print(f"outliers (topic -1): {(df.topic == -1).mean():.1%}")
    return model, df, tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--min-topic-size", type=int, default=15)
    a = ap.parse_args()
    run(a.n, a.min_topic_size)
