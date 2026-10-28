"""Tests for the ML pipeline.

Run with:  pytest -q
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.dataset import normalise
from src.preprocessing import clean_text


# ------------------------------------------------------------------ cleaning
def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, WORLD!!!") == "hello world"


def test_clean_text_replaces_urls_emails_and_numbers():
    result = clean_text("Win £1000! Mail me at a.b@c.com or see http://x.co")
    assert "__money__" in result
    assert "__email__" in result
    assert "__url__" in result
    assert "£" not in result and "@" not in result


def test_clean_text_detects_phone_numbers_with_spaces():
    assert clean_text("Call 0800 123 4567 now") == "call __phone__ now"


def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(12345) == "__number__"


def test_clean_text_contracts_apostrophes():
    assert clean_text("Don't stop") == "dont stop"


# ------------------------------------------------------------------- dataset
def test_normalise_filters_labels_and_drops_duplicates():
    raw = pd.DataFrame(
        {
            "v1": ["ham", "spam", "HAM", "junk"],
            "v2": ["see you at six", "win cash now", "see you at six", "not a real label"],
        }
    )
    clean = normalise(raw)

    assert list(clean.columns) == ["label", "message"]
    assert set(clean["label"]) == {"ham", "spam"}
    assert len(clean) == 2  # duplicate message + invalid label removed


def test_normalise_accepts_headerless_frames():
    raw = pd.DataFrame({0: ["ham", "spam"], 1: ["hello there", "free money"]})
    assert len(normalise(raw)) == 2


# ------------------------------------------------------------------ training
def test_training_and_prediction_roundtrip(synthetic_dataset, tmp_path):
    from src import train as train_module
    from src.model import save_model

    with patch.object(train_module, "ensure_dataset", return_value=synthetic_dataset):
        result = train_module.train(
            test_size=0.25, seed=0, cv_folds=2, n_jobs=1, save=False, verbose=False
        )

    classifier = result["classifier"]
    assert classifier.predict_one("congratulations you won cash call now") == "spam"
    assert classifier.predict_one("see you at lunch tomorrow") == "ham"
    assert classifier.version != "unknown"

    probabilities = classifier.spam_probability(["free cash prize", "lunch tomorrow"])
    assert probabilities.shape == (2,)
    assert probabilities[0] > probabilities[1]

    path = tmp_path / "model.joblib"
    save_model(classifier, path)
    reloaded = type(classifier).load(path)
    assert reloaded.predict_one("you won a free prize") == "spam"
    assert reloaded.version != classifier.version or len(reloaded.version) == 12


def test_explain_returns_ranked_tokens(trained_classifier):
    tokens = trained_classifier.explain("free cash prize call now", top_k=5)
    assert tokens
    assert len(tokens) <= 5
    assert {"token", "weight"} <= set(tokens[0])
    assert tokens[0]["weight"] > 0


@pytest.mark.parametrize("text", ["", "   ", "!!!", "12345"])
def test_prediction_handles_degenerate_input(trained_classifier, text):
    assert trained_classifier.predict_one(text) in {"ham", "spam"}
