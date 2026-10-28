"""Text normalisation shared by training and inference.

The same function must be used in both places, otherwise the model sees a
different vocabulary at prediction time than it was trained on.
"""

from __future__ import annotations

import html
import re

# Order matters: URLs/emails/money/phones are replaced *before* the generic
# digit rule so that the model still gets a useful signal from them.
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_MONEY = re.compile(r"[£$€]\s?\d+(?:[.,]\d+)?")
_PHONE = re.compile(r"\+?\d[\d\s\-().]{6,}\d")
_DIGITS = re.compile(r"\d+")
_APOSTROPHE = re.compile(r"['’]")
_NON_LETTERS = re.compile(r"[^a-z_\s]")
_SPACES = re.compile(r"\s+")

# Tokens that replace the removed patterns. They are made of letters and
# underscores so scikit-learn's default tokenizer keeps them intact.
URL_TOKEN = "__url__"
EMAIL_TOKEN = "__email__"
MONEY_TOKEN = "__money__"
PHONE_TOKEN = "__phone__"
NUMBER_TOKEN = "__number__"


def _replace_phone(match: re.Match) -> str:
    """Only treat a run of digits/separators as a phone number with 7+ digits."""
    candidate = match.group(0)
    if sum(character.isdigit() for character in candidate) >= 7:
        return f" {PHONE_TOKEN} "
    return candidate


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and replace high-signal patterns.

    ``"Win £1000! Call 0800 123 4567 now -> http://x.co"`` becomes
    ``"win __money__ call __phone__ now __url__"``.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = html.unescape(text).lower()
    text = _URL.sub(f" {URL_TOKEN} ", text)
    text = _EMAIL.sub(f" {EMAIL_TOKEN} ", text)
    text = _MONEY.sub(f" {MONEY_TOKEN} ", text)
    text = _PHONE.sub(_replace_phone, text)
    text = _DIGITS.sub(f" {NUMBER_TOKEN} ", text)
    text = _APOSTROPHE.sub("", text)          # don't -> dont
    text = _NON_LETTERS.sub(" ", text)        # drop punctuation / symbols
    text = _SPACES.sub(" ", text)
    return text.strip()
