from src.clean import clean_text, extract_mentions

def test_clean_strips_urls_and_mentions():
    out = clean_text("@bob look http://x.co/a &amp; this")
    assert "http" not in out and "@" not in out and "&" in out

def test_emoji_demojize():
    assert "grinning" in clean_text("hi \U0001F600")

def test_mentions():
    assert extract_mentions("@Bob hi @alice_1") == ["bob", "alice_1"]
