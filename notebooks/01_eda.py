# %% [markdown]
# # 01 - Exploratory data analysis
# Open in VS Code: each `# %%` block runs as a notebook cell (Shift+Enter).
# Run VS Code from the project root so `src` imports work.

# %%
import sys; sys.path.append(".")
import pandas as pd
from src.load_data import *
from src.clean import clean_text, extract_mentions

# %%
posts, comments = load_reddit_posts(), load_reddit_comments()
print(posts.shape, comments.shape, posts.created.min(), posts.created.max())
posts.groupby(posts.created.dt.year).size().plot(kind="bar", title="Reddit posts per year")

# %%
s140 = load_sentiment140(n=200_000)   # sample while developing
s140["mentions"] = s140.text.map(extract_mentions)
print(s140.label.value_counts(), (s140.mentions.str.len() > 0).mean())

# %%
liar = load_liar("train", binary=True)
print(liar.label.value_counts(), liar.label_bin.value_counts())

# %%
yt = load_youtube()
yt.groupby("content_category")[["ctr_percentage", "likes"]].mean()
