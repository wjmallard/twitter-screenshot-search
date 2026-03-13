"""OCR text cleaning — remove garbage lines from Tesseract output.

Tesseract interprets UI chrome (avatars, blue checks, engagement bars,
retweet/like icons) as short gibberish tokens.  These are harmless for
keyword search, but they poison embedding models.

Heuristic: a line is kept only if it contains at least one "word-like"
token — 2+ characters, majority alphabetic.  This filters lines like
'Q 47 tl 238 @& 1.2K' (engagement bar) or '© |' (icons) while keeping
real tweet content even if it contains short words or the occasional
gibberish token mixed in.
"""


def _is_word_like(token: str) -> bool:
    return len(token) >= 2 and sum(c.isalpha() for c in token) > len(token) / 2


def clean_ocr_text(text: str) -> str:
    """Return text with garbage lines removed and whitespace normalized."""
    cleaned = []
    for line in text.splitlines():
        tokens = line.split()
        if any(_is_word_like(t) for t in tokens):
            cleaned.append(" ".join(tokens))
    return " ".join(cleaned)
