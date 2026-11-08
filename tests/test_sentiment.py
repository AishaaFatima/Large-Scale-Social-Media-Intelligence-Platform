from src.sentiment import preprocess

def test_preprocess_placeholders():
    assert preprocess("@bob see https://x.co/a now") == "@user see http now"

def test_preprocess_non_string():
    assert preprocess(None) == ""
