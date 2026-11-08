# %% [markdown]
# # 02 - Sentiment over time
# Run first:  python -m src.sentiment --dataset reddit_comments --n 5000
#             python -m src.sentiment --benchmark --n 2000

# %%
import sys; sys.path.append(".")
import pandas as pd, matplotlib.pyplot as plt
from src.config import PROCESSED


# %%
df = pd.read_parquet(PROCESSED / "reddit_comments_sentiment.parquet")
print(df.label.value_counts(normalize=True))

# %% [markdown]
# ## Transformer vs the dataset's built-in (VADER-style) score

# %%
print("correlation:", df[["sent_score", "vader_sentiment"]].corr().iloc[0, 1])
df.plot.scatter("vader_sentiment", "sent_score", alpha=0.2, title="Transformer vs VADER")

# %% [markdown]
# ## Monthly trend

# %%
m = df.set_index("created").resample("QS")["sent_score"].agg(["mean", "count"])
m = m[m["count"] >= 10]
ax = m["mean"].plot(title="Mean transformer sentiment per quarter (Reddit comments)")
ax.axhline(0, color="gray", lw=0.5)
plt.show()

# %% [markdown]
# ## Benchmark on Sentiment140

# %%
pd.read_csv(PROCESSED / "benchmark_sentiment140_roberta.csv")
