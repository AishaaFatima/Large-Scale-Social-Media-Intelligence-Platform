"""User-interaction graph from Sentiment140 @mentions (step 5).

Usage (project root, venv active):
    python -m src.graph --n 200000            # quick test on a random sample of tweets
    python -m src.graph                       # all 1.6M tweets
    python -m src.graph --neo4j               # also load the core graph into Neo4j (needs .env)
Edge A -> B means user A mentioned user B (weight = number of mentions).
Outputs in data/processed/: edges.parquet, nodes.parquet (pagerank, community, ...)
"""
import argparse
import os

import pandas as pd

from .config import PROCESSED, SEED
from .load_data import load_sentiment140


def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tweets (columns: user, text) into weighted user->mentioned_user edges."""
    mentions = df["text"].str.lower().str.findall(r"@(\w+)")
    e = pd.DataFrame({"src": df["user"].str.lower().values, "dst": mentions.values})
    e = e.explode("dst").dropna(subset=["dst"])
    e = e[e["src"] != e["dst"]]
    return e.groupby(["src", "dst"], as_index=False).size().rename(columns={"size": "weight"})


def core_edges(edges: pd.DataFrame, min_component: int = 5) -> pd.DataFrame:
    """Drop tiny disconnected clusters (isolated pairs, etc.) but keep stars: users mentioning a hub stay."""
    import networkx as nx

    U = nx.Graph()
    U.add_edges_from(zip(edges["src"], edges["dst"]))
    keep = set()
    for comp in nx.connected_components(U):
        if len(comp) >= min_component:
            keep |= comp
    return edges[edges["src"].isin(keep) & edges["dst"].isin(keep)]


def analyse(edges: pd.DataFrame, tweets: pd.DataFrame, min_component: int = 5) -> pd.DataFrame:
    """PageRank on the full directed graph; Louvain communities on the undirected core."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges.itertuples(index=False, name=None))
    pr = nx.pagerank(G, alpha=0.85, weight="weight")

    core = core_edges(edges, min_component)
    U = nx.Graph()
    U.add_weighted_edges_from(core.itertuples(index=False, name=None))
    comms = sorted(nx.community.louvain_communities(U, weight="weight", seed=SEED), key=len, reverse=True)
    comm_of = {u: i for i, c in enumerate(comms) for u in c}

    nodes = pd.DataFrame({"user": list(pr.keys()), "pagerank": list(pr.values())})
    nodes["in_degree"] = nodes["user"].map(dict(G.in_degree()))
    nodes["out_degree"] = nodes["user"].map(dict(G.out_degree()))
    nodes["mentions_received"] = nodes["user"].map(dict(G.in_degree(weight="weight")))
    nodes["community"] = nodes["user"].map(comm_of).fillna(-1).astype(int)   # -1 = not in core

    t = tweets.assign(user=tweets["user"].str.lower(), pos=(tweets["label"] == "positive"))
    stats = t.groupby("user").agg(n_tweets=("pos", "size"), pos_rate=("pos", "mean")).reset_index()
    nodes = nodes.merge(stats, on="user", how="left")
    return nodes.sort_values("pagerank", ascending=False).reset_index(drop=True)


def select_for_neo4j(nodes: pd.DataFrame, edges: pd.DataFrame,
                     max_nodes: int = 150_000, max_rels: int = 250_000):
    """Pick whole communities (largest first) until the budget is used, so AuraDB Free limits are respected."""
    n = nodes[nodes["community"] >= 0]
    comm = n.set_index("user")["community"]
    e = edges[edges["src"].isin(comm.index) & edges["dst"].isin(comm.index)].copy()
    e["c_src"], e["c_dst"] = e["src"].map(comm), e["dst"].map(comm)
    rels = e[e["c_src"] == e["c_dst"]].groupby("c_src").size()
    chosen, tn, tr = [], 0, 0
    for c, size in n["community"].value_counts().items():      # largest community first
        r = int(rels.get(c, 0))
        if tn + size > max_nodes or tr + r > max_rels:
            continue                                            # too big for what is left; try smaller ones
        chosen.append(c)
        tn, tr = tn + size, tr + r
    keep = n[n["community"].isin(chosen)]
    users = set(keep["user"])
    ke = e[e["src"].isin(users) & e["dst"].isin(users)][["src", "dst", "weight"]]
    cap = int(max_rels * 1.4)                                   # cross-community edges between chosen ones
    if len(ke) > cap:
        ke = ke.nlargest(cap, "weight")
    return keep, ke


def load_neo4j(nodes: pd.DataFrame, edges: pd.DataFrame, max_nodes: int = 150_000,
               max_rels: int = 250_000, batch: int = 5000):
    """Load a size-capped subgraph into Neo4j (works for AuraDB and local). Safe to re-run (MERGE)."""
    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv()
    drv = GraphDatabase.driver(os.environ["NEO4J_URI"],
                               auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    drv.verify_connectivity()
    cn, ce = select_for_neo4j(nodes, edges, max_nodes, max_rels)
    cn = cn.fillna(0)
    print(f"loading {len(cn)} users and {len(ce)} MENTIONS edges "
          f"({cn['community'].nunique()} communities) into Neo4j")

    with drv.session() as s:
        s.run("CREATE CONSTRAINT user_name IF NOT EXISTS FOR (u:User) REQUIRE u.name IS UNIQUE")
        rows = cn.rename(columns={"user": "name"}).to_dict("records")
        for i in range(0, len(rows), batch):
            s.run("UNWIND $rows AS r MERGE (u:User {name: r.name}) "
                  "SET u.pagerank = r.pagerank, u.community = r.community, u.in_degree = r.in_degree, "
                  "u.n_tweets = r.n_tweets, u.pos_rate = r.pos_rate", rows=rows[i:i + batch])
            print(f"  nodes {min(i + batch, len(rows))}/{len(rows)}")
        erows = ce.to_dict("records")
        for i in range(0, len(erows), batch):
            s.run("UNWIND $rows AS r MATCH (a:User {name: r.src}), (b:User {name: r.dst}) "
                  "MERGE (a)-[m:MENTIONS]->(b) SET m.weight = r.weight", rows=erows[i:i + batch])
            print(f"  edges {min(i + batch, len(erows))}/{len(erows)}")
    drv.close()
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="random sample of tweets (default: all)")
    ap.add_argument("--min-component", type=int, default=5)
    ap.add_argument("--neo4j", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=150_000, help="Neo4j node budget (Aura Free cap: 200k)")
    ap.add_argument("--max-rels", type=int, default=250_000, help="Neo4j relationship budget (Aura Free cap: 400k)")
    a = ap.parse_args()

    tweets = load_sentiment140()
    if a.n:
        tweets = tweets.sample(a.n, random_state=SEED)
    edges = build_edges(tweets)
    print(f"{len(edges)} unique edges")
    nodes = analyse(edges, tweets, a.min_component)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(PROCESSED / "edges.parquet")
    nodes.to_parquet(PROCESSED / "nodes.parquet")

    sizes = nodes[nodes.community >= 0].community.value_counts()
    print(f"{len(nodes)} users | {len(sizes)} communities | largest sizes: {sizes.head(5).tolist()}")
    print(nodes.head(15)[["user", "pagerank", "in_degree", "mentions_received", "community", "pos_rate"]].to_string(index=False))
    if a.neo4j:
        load_neo4j(nodes, edges, a.max_nodes, a.max_rels)
