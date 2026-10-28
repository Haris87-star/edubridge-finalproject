"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


def build_synthetic_dataset(rows: int = 60) -> pd.DataFrame:
    """Small but learnable dataset so tests never hit the network."""
    spam = [
        "congratulations you won a cash prize call now to claim free money",
        "urgent your account is suspended verify your details win a reward",
        "free entry to the competition text win now prize cash bonus",
        "claim your guaranteed loan approved reply now cash offer",
    ]
    ham = [
        "hey are we still meeting for lunch tomorrow at one",
        "ok see you at the station in twenty minutes",
        "can you send me the notes from the lecture please",
        "i am running late but will be there soon",
    ]
    records = []
    for index in range(rows):
        records.append({"label": "spam", "message": f"{spam[index % len(spam)]} {index}"})
        records.append({"label": "ham", "message": f"{ham[index % len(ham)]} {index}"})
    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def synthetic_dataset() -> pd.DataFrame:
    return build_synthetic_dataset()


@pytest.fixture(scope="session")
def trained_classifier(synthetic_dataset):
    """Train once for the whole session (~2s) and reuse everywhere."""
    from src import train as train_module

    with patch.object(train_module, "ensure_dataset", return_value=synthetic_dataset):
        result = train_module.train(
            test_size=0.25,
            seed=0,
            cv_folds=2,
            n_jobs=1,
            save=False,
            verbose=False,
        )
    return result["classifier"]
