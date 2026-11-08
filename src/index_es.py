"""Index everything into Elasticsearch (Elastic Cloud) for Kibana dashboards and search (step 7).

Setup: put these in .env (never commit it):
    ELASTIC_URL=https://<your-endpoint>.elastic.cloud:443
    ELASTIC_API_KEY=<api key>
Usage (project root, venv active):
    python -m src.index_es --check                      # test the connection
    python -m src.index_es --all                        # create indices + load everything
    python -m src.index_es --only claims users          # load just some indices
    python -m src.index_es --query "election"           # quick full-text search demo
Re-running is safe: documents use stable _ids, so they are overwritten, not duplicated.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

from .config import PROCESSED, ROOT, SEED
from .load_data import load_liar, load_sentiment140, load_youtube

MAPPINGS = {
    "reddit-comments": {"id": "keyword", "post_id": "keyword", "created": "date", "text": "text",
                        "label": "keyword", "sent_score": "float", "p_negative": "float", "p_neutral": "float",
                        "p_positive": "float", "vader_sentiment": "float", "score": "integer"},
    "reddit-posts": {"id": "keyword", "created": "date", "title": "text", "text": "text", "domain": "keyword",
                     "score": "integer", "topic": "integer", "topic_name": "keyword"},
    "tweets": {"tweet_id": "keyword", "user": "keyword", "created": "date", "text": "text", "label": "keyword",
               "pagerank": "float", "community": "integer"},
    "users": {"user": "keyword", "pagerank": "float", "in_degree": "integer", "out_degree": "integer",
              "mentions_received": "integer", "community": "integer", "n_tweets": "integer", "pos_rate": "float"},
    "claims": {"id": "keyword", "split": "keyword", "statement": "text", "speaker": "keyword", "party": "keyword",
               "subject": "keyword", "context": "text", "label": "keyword", "label_bin": "keyword",
               "pred_bin": "keyword", "prob_fake": "float", "correct": "boolean"},
    "youtube-videos": {"post_id": "keyword", "upload_date": "date", "video_duration_min": "float",
                       "avg_view_duration_sec": "integer", "avg_view_percentage": "float",
                       "subscribers_gained": "integer", "traffic_source": "keyword", "ctr_percentage": "float",
                       "impressions": "long", "likes": "integer", "comments": "integer", "shares": "integer",
                       "total_watch_time_hours": "float", "content_category": "keyword"},
}


def get_client():
    from dotenv import load_dotenv
    from elasticsearch import Elasticsearch

    load_dotenv()
    return Elasticsearch(os.environ["ELASTIC_URL"], api_key=os.environ["ELASTIC_API_KEY"], request_timeout=120)


def df_to_actions(df: pd.DataFrame, index: str, id_col: str):
    """Yield bulk actions: datetimes -> ISO strings, NaN -> None, stable _id."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df = df.astype(object).where(df.notna(), None)
    for rec in df.to_dict("records"):
        yield {"_index": index, "_id": str(rec[id_col]), "_source": rec}


# ---- one prepare_* function per index (pure pandas, testable without Elasticsearch) ----

def prepare_comments() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "reddit_comments_sentiment.parquet")
    return df[["id", "post_id", "created", "text", "label", "sent_score", "p_negative", "p_neutral",
               "p_positive", "vader_sentiment", "score"]]


def prepare_posts() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "post_topics.parquet")
    info = pd.read_csv(PROCESSED / "topic_info.csv").set_index("Topic")["Name"]
    df["topic_name"] = df["topic"].map(info).fillna("outlier")
    return df[["id", "created", "title", "text", "domain", "score", "topic", "topic_name"]]


def prepare_users(top: int = 50_000) -> pd.DataFrame:
    nodes = pd.read_parquet(PROCESSED / "nodes.parquet")
    return nodes.head(top)[["user", "pagerank", "in_degree", "out_degree", "mentions_received",
                            "community", "n_tweets", "pos_rate"]]


def prepare_tweets(n: int = 50_000) -> pd.DataFrame:
    tw = load_sentiment140().groupby("label", group_keys=False).sample(n // 2, random_state=SEED)
    nodes = pd.read_parquet(PROCESSED / "nodes.parquet")[["user", "pagerank", "community"]]
    tw["user"] = tw["user"].str.lower()
    tw = tw.merge(nodes, on="user", how="left")
    tw["tweet_id"] = tw["tweet_id"].astype(str)
    return tw[["tweet_id", "user", "created", "text", "label", "pagerank", "community"]]


def prepare_claims() -> pd.DataFrame:
    """All LIAR splits with the trained binary classifier's prediction. (train/valid are in-sample: filter split:test.)"""
    pipe = joblib.load(ROOT / "models" / "liar_label_bin.joblib")
    fake_i = list(pipe.classes_).index("fake")
    parts = []
    for split in ("train", "valid", "test"):
        d = load_liar(split, binary=True)
        d["split"] = split
        d["id"] = split + "_" + d["id"].astype(str)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    proba = pipe.predict_proba(df["statement"])
    df["prob_fake"] = proba[:, fake_i]
    df["pred_bin"] = np.where(df["prob_fake"] >= 0.5, "fake", "real")
    df["correct"] = df["pred_bin"] == df["label_bin"]
    return df[["id", "split", "statement", "speaker", "party", "subject", "context", "label", "label_bin",
               "pred_bin", "prob_fake", "correct"]]


def prepare_youtube() -> pd.DataFrame:
    return load_youtube()


PREPARE = {"reddit-comments": (prepare_comments, "id"), "reddit-posts": (prepare_posts, "id"),
           "tweets": (prepare_tweets, "tweet_id"), "users": (prepare_users, "user"),
           "claims": (prepare_claims, "id"), "youtube-videos": (prepare_youtube, "post_id")}
ALIASES = {"comments": "reddit-comments", "posts": "reddit-posts", "youtube": "youtube-videos"}


def load_index(es, name: str, recreate: bool = False):
    from elasticsearch import helpers

    if recreate and es.indices.exists(index=name):
        es.indices.delete(index=name)
    if not es.indices.exists(index=name):
        es.indices.create(index=name, mappings={"properties": {k: {"type": v} for k, v in MAPPINGS[name].items()}})
    fn, id_col = PREPARE[name]
    df = fn()
    ok, errs = helpers.bulk(es, df_to_actions(df, name, id_col), chunk_size=2000, raise_on_error=False)
    print(f"{name}: indexed {ok} docs" + (f", {len(errs)} errors, e.g. {errs[0]}" if errs else ""))


def search(es, q: str, size: int = 3):
    for idx, fields in {"reddit-posts": ["title", "text"], "reddit-comments": ["text"],
                        "tweets": ["text"], "claims": ["statement", "context"]}.items():
        if not es.indices.exists(index=idx):
            continue
        r = es.search(index=idx, size=size, query={"simple_query_string": {"query": q, "fields": fields}})
        print(f"\n[{idx}] {r['hits']['total']['value']} hits")
        for h in r["hits"]["hits"]:
            s = h["_source"]
            print("  -", (s.get("title") or s.get("text") or s.get("statement") or "")[:110].replace("\n", " "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="+", help="comments posts tweets users claims youtube")
    ap.add_argument("--recreate", action="store_true", help="delete and recreate the index first")
    ap.add_argument("--query")
    a = ap.parse_args()
    es = get_client()
    if a.check:
        print("connected:", es.info()["version"]["number"] if hasattr(es, "info") else "ok")
    if a.all or a.only:
        for name in (list(PREPARE) if a.all else [ALIASES.get(x, x) for x in a.only]):
            load_index(es, name, a.recreate)
    if a.query:
        search(es, a.query)
