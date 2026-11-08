# Large-Scale Social Media Intelligence Platform

An end-to-end pipeline that loads social-media datasets, measures **sentiment over time**, discovers **topics**, maps **influencers** with a user-interaction graph, flags likely **misinformation**, and exposes everything through **Elasticsearch + Kibana** (search, dashboards, alerts).

<img width="1180" height="740" alt="architecture" src="https://github.com/user-attachments/assets/13d84a41-0ca1-4a88-a733-d171d055e381" />


## Results at a glance

| Component | Model / method | Data | Result |
|---|---|---|---|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment140, 4,000 balanced tweets | **77.2 % accuracy**, F1 0.787 (positive) / 0.755 (negative) |
| Sentiment over time | same model | 5,000 random Reddit comments | 62.4 % neutral, 24.7 % positive, 12.9 % negative; correlation with the dataset's VADER-style score **0.51**; flat trend 2013-2022 (mean +0.05 to +0.28) |
| Topics | BERTopic (MiniLM embeddings, UMAP, HDBSCAN) | 20,160 Reddit posts | 32.3 % outliers. Largest topics: COVID, elections, crime/police, schools, weather, music, movies, Twitter data, healthcare, real estate, games, open data, retail. **TODO: add total topic count** (`len(topic_info.csv) - 1`) |
| Influence graph | NetworkX PageRank + Louvain communities | Sentiment140 `@mentions` | 637,415 weighted edges, 534,347 users, 3,863 communities (largest 27,442). Top PageRank: `mileycyrus`, `ddlovato`, `tommcfly`, `jonasbrothers`, `taylorswift13` |
| Misinformation (binary) | TF-IDF + Logistic Regression | LIAR test split (1,283) | **61.7 % accuracy**, macro-F1 0.612 (chance about 50 %) |
| Misinformation (6 labels) | same | LIAR test split | **25.1 % accuracy**, macro-F1 0.239 (chance about 17 %) |

Sentiment140's labels come from emoticons, so they are noisy; about 77 % is in the normal range for a general-purpose model on that data. LIAR labels depend on fact-checking, not writing style, so text-only models stay low; the 6-class number is expected.

## Datasets

Place these in `data/raw/` (not committed). The brief allows offline datasets; no live API collection was used.

| File | Rows | Used for |
|---|---|---|
| `reddit_posts.csv` | 20,292 (20,160 used after dropping deleted/short posts) | topics over time, search |
| `reddit_comments.csv` | 54,848 | sentiment over time |
| `training.1600000.processed.noemoticon.csv` (Sentiment140) | 1,600,000 | sentiment benchmark, @mention graph |
| `train.tsv`, `valid.tsv`, `test.tsv` (LIAR) | 10,269 / 1,284 / 1,283 | misinformation classifier |
| `YouTube_Video.csv` | 29,999 | engagement dashboard |

Known data limits: the Reddit export covers a single subreddit (`r/datasets`) and has no author field, so the user graph comes from Sentiment140 instead. Sentiment140 has only positive/negative labels in the training file. No dataset here is meaningfully multilingual, so multilingual sentiment is out of scope.

## Setup (Windows 11, VS Code, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy pyarrow tqdm emoji langdetect pytest matplotlib ipykernel joblib
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-learn bertopic sentence-transformers umap-learn
pip install networkx scipy neo4j python-dotenv "elasticsearch>=8.15"
pytest
```

Copy `.env.example` to `.env` and fill in the Neo4j AuraDB and Elastic Cloud credentials. Never commit `.env`.

## Running the pipeline

```powershell
python -m src.sentiment --dataset reddit_comments --n 5000   # sentiment on Reddit comments
python -m src.sentiment --benchmark --n 2000                 # Sentiment140 accuracy / F1
python -m src.topics                                         # BERTopic + topics over time
python -m src.graph --neo4j                                  # edges, PageRank, communities, load to Neo4j
python -m src.misinfo                                        # LIAR classifier (saves models/*.joblib)
python -m src.index_es --all                                 # index into Elasticsearch
python -m src.index_es --query "election"                    # full-text search demo
```

Notebooks (`notebooks/*.py`) use `# %%` cells and run directly in VS Code: `01_eda.py`, `02_sentiment_over_time.py`.

## Data and API documentation

**Code layout**

| Module | Purpose |
|---|---|
| `src/config.py` | paths, file names, random seed |
| `src/load_data.py` | one loader per dataset, consistent column names |
| `src/clean.py` | `clean_text`, `extract_mentions`, `detect_language` |
| `src/sentiment.py` | `predict()`, `run_dataset()`, `benchmark_sentiment140()` |
| `src/topics.py` | BERTopic fit, topics over time, HTML chart in `docs/` |
| `src/graph.py` | `build_edges`, `analyse` (PageRank, Louvain), `load_neo4j` |
| `src/misinfo.py` | TF-IDF + LogReg, hyper-parameter chosen on validation, metrics on test |
| `src/index_es.py` | index mappings, bulk loading, search demo |

**Neo4j model:** `(:User {name, pagerank, community, in_degree, n_tweets, pos_rate})-[:MENTIONS {weight}]->(:User)`. Example: top influencers
```cypher
MATCH (u:User) RETURN u.name, u.pagerank, u.community ORDER BY u.pagerank DESC LIMIT 10
```

**Elasticsearch indices:** `reddit-comments`, `reddit-posts`, `tweets` (50k balanced sample), `users` (top 50k by PageRank), `claims` (LIAR with predicted `prob_fake`), `youtube-videos`. Field types are in `MAPPINGS` in `src/index_es.py`. Always filter `claims` on `split:test`: predictions on train/valid are in-sample (about 75 % vs 61.7 % on test).

Dashboard build steps: [`docs/kibana_dashboards.md`](docs/kibana_dashboards.md).

## Design decisions and deviations from the brief

- **Hugging Face transformers instead of Spark NLP** for sentiment. Spark NLP needs Java and Hadoop setup on Windows; transformers gave the same task with less risk. Spark NLP is not used.
- **Offline datasets instead of live Reddit/YouTube API collection** (the brief lists offline datasets as alternatives).
- **Neo4j on AuraDB Free**, which caps data size, so only the largest communities (up to 150,000 users / 250,000 relationships) are loaded. The full graph and all rankings are in `data/processed/*.parquet`.
- **Sentiment on a 5,000-comment random sample** of Reddit comments (CPU inference, roughly 18 texts/s), not all 54,848.
- **Language detection** is implemented (`detect_language`) but was not run at scale; the data is almost entirely English.
- **Elastic Cloud trial** is time-limited (14 days), so dashboard screenshots and a screen recording are included as evidence. **TODO: add them to `docs/`.**

## Challenges worth noting

- Reddit export has no usernames, which forced the graph onto Sentiment140.
- A column-name clash (`score` for upvotes vs model score) and a CSV quoting bug in LIAR (rows silently merged) were found and fixed; the LIAR loader now uses `quoting=csv.QUOTE_NONE`.
- BERTopic's keyword step failed on small samples until `min_df` was lowered; `[deleted]` / `[removed]` posts form fake topics and are filtered out.

## Next steps

Fine-tune a transformer on LIAR, score all Reddit comments, add speaker/party metadata features, ingest live API data, and add scheduled alert rules.

## Tests

`pytest` runs unit tests for cleaning, preprocessing, edge building, community selection, the LIAR pipeline, and Elasticsearch action conversion.
