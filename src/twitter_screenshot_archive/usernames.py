"""Twitter username extraction from OCR text."""

import re

_USERNAME_RE = re.compile(r"@([A-Za-z0-9_]{1,15})\b")

# Common OCR false positives: single chars, digits, and short English words
# that appear after @-like artifacts. Real handles like @ap, @un are kept.
_USERNAME_BLACKLIST = frozenset(
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + [
        "an",
        "as",
        "bo",
        "by",
        "dd",
        "ft",
        "is",
        "ma",
        "me",
        "mo",
        "mr",
        "no",
        "re",
        "se",
        "the",
    ]
)


def extract_usernames(text: str) -> list[str]:
    """Extract Twitter usernames from text, order-preserving dedup, lowercased."""
    seen = {}
    for match in _USERNAME_RE.finditer(text):
        name = match.group(1).lower()
        if name not in seen and name not in _USERNAME_BLACKLIST:
            seen[name] = True
    return list(seen)
