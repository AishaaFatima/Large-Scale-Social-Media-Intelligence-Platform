import pandas as pd
import pytest
from src.graph import build_edges, core_edges

def _tweets():
    return pd.DataFrame({
        "user": ["Alice", "alice", "bob", "carol"],
        "text": ["hi @Bob and @bob", "@bob again", "@alice thanks @carol", "no mention, @carol self"],
    })

def test_build_edges_weights_and_no_self_loops():
    e = build_edges(_tweets()).set_index(["src", "dst"])["weight"].to_dict()
    assert e[("alice", "bob")] == 3
    assert ("carol", "carol") not in e

def test_core_edges_keeps_stars_drops_isolated_pairs():
    pytest.importorskip("networkx")
    e = pd.DataFrame({"src": ["f1", "f2", "f3", "f4", "x"], "dst": ["hub", "hub", "hub", "hub", "y"], "weight": 1})
    c = core_edges(e, 5)
    assert len(c) == 4 and "x" not in set(c.src)


def test_select_for_neo4j_respects_budget():
    from src.graph import select_for_neo4j
    nodes = pd.DataFrame({"user": list("abcdef"), "community": [0, 0, 0, 1, 1, -1]})
    edges = pd.DataFrame({"src": ["a", "b", "d"], "dst": ["b", "c", "e"], "weight": 1})
    keep, ke = select_for_neo4j(nodes, edges, max_nodes=3, max_rels=10)
    assert set(keep.user) == {"a", "b", "c"} and len(ke) == 2
