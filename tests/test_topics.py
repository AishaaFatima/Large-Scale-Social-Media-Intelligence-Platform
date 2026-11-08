import pandas as pd
from src.topics import prepare_docs

def test_prepare_docs_filters_short_and_strips_urls():
    df = pd.DataFrame({"text": ["hi", "check https://x.co/a this long enough dataset question", None]})
    out = prepare_docs(df)
    assert len(out) == 1 and "http" not in out.doc.iloc[0]
