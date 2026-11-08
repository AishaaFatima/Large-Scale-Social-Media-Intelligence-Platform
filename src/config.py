"""Central paths and dataset file names. Put raw files in data/raw/ using these names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "reddit_posts": RAW / "reddit_posts.csv",
    "reddit_comments": RAW / "reddit_comments.csv",
    "youtube": RAW / "YouTube_Video.csv",
    "sentiment140": RAW / "training.1600000.processed.noemoticon.csv",
    "liar_train": RAW / "train.tsv",
    "liar_valid": RAW / "valid.tsv",
    "liar_test": RAW / "test.tsv",
}

SEED = 42
