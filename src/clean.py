"""Text cleaning: URLs, mentions, HTML entities, emojis, whitespace; language detection."""
import html
import re
import emoji

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")


def extract_mentions(text: str) -> list[str]:
    """Lower-cased @mentions, used to build the user-interaction graph."""
    return [m.lower() for m in re.findall(r"@(\w+)", text or "")]


def clean_text(text: str, keep_mentions: bool = False, emoji_mode: str = "demojize") -> str:
    """emoji_mode: 'demojize' -> 'grinning face' (keeps sentiment signal), 'remove', or 'keep'."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _URL.sub(" ", t)
    if not keep_mentions:
        t = _MENTION.sub(" ", t)
    if emoji_mode == "demojize":
        t = emoji.demojize(t, delimiters=(" ", " ")).replace("_", " ")
    elif emoji_mode == "remove":
        t = emoji.replace_emoji(t, "")
    return _WS.sub(" ", t).strip()


def detect_language(text: str) -> str:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text) if text and len(text) > 3 else "unknown"
    except Exception:
        return "unknown"
