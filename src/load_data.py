"""Loaders that return clean, consistently named DataFrames for every dataset."""
import csv

import pandas as pd
from .config import FILES

S140_COLS = ["sentiment", "tweet_id", "date", "query", "user", "text"]
LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
             "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context"]
LIAR_FAKE = {"pants-fire", "false", "barely-true"}


def load_reddit_posts() -> pd.DataFrame:
    df = pd.read_csv(FILES["reddit_posts"])
    df = df.rename(columns={"subreddit.name": "subreddit"})
    df["created"] = pd.to_datetime(df["created_utc"], unit="s")
    df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()
    return df[["id", "subreddit", "created", "domain", "score", "title", "selftext", "text", "permalink"]]


def load_reddit_comments() -> pd.DataFrame:
    df = pd.read_csv(FILES["reddit_comments"])
    df = df.rename(columns={"subreddit.name": "subreddit", "body": "text",
                            "sentiment": "vader_sentiment"})
    df["created"] = pd.to_datetime(df["created_utc"], unit="s")
    # permalink: .../comments/<post_id>/<slug>/<comment_id>/  -> link comment to its post
    df["post_id"] = df["permalink"].str.extract(r"/comments/([a-z0-9]+)/")
    df = df.dropna(subset=["text"])
    return df[["id", "post_id", "subreddit", "created", "text", "vader_sentiment", "score", "permalink"]]


def load_sentiment140(n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(FILES["sentiment140"], encoding="ISO-8859-1", names=S140_COLS, nrows=n)
    # raw labels: 0 = negative, 4 = positive (2 = neutral only in the small manual test file)
    df["label"] = df["sentiment"].map({0: "negative", 2: "neutral", 4: "positive"})
    # "Mon Apr 06 22:19:45 PDT 2009" -> drop the tz token, which pandas cannot parse
    df["created"] = pd.to_datetime(df["date"].str.replace(r" P[SD]T", "", regex=True),
                                   format="%a %b %d %H:%M:%S %Y")
    return df[["tweet_id", "user", "created", "text", "label"]]


def load_liar(split: str = "train", binary: bool = False) -> pd.DataFrame:
    df = pd.read_csv(FILES[f"liar_{split}"], sep="\t", header=None, names=LIAR_COLS,
                     quoting=csv.QUOTE_NONE)  # statements contain stray quotes; default parsing merges rows
    if binary:
        df["label_bin"] = df["label"].map(lambda x: "fake" if x in LIAR_FAKE else "real")
    return df


def load_youtube() -> pd.DataFrame:
    df = pd.read_csv(FILES["youtube"])
    df["upload_date"] = pd.to_datetime(df["upload_date"])
    return df
