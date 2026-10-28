"""Model lifecycle and prediction logic, independent of the HTTP layer."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import SPAM
from src.model import SpamClassifier

from .settings import Settings


class ModelNotReady(RuntimeError):
    """Raised when a prediction is attempted before a model is available."""


class ModelService:
    """Owns the loaded classifier and reloads it safely.

    Kept free of FastAPI imports so it can be tested (and reused) directly.
    """

    def __init__(self, settings: Settings, classifier: SpamClassifier | None = None):
        self.settings = settings
        self._classifier = classifier
        self._loaded_at = datetime.now(timezone.utc) if classifier else None
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- loading
    @property
    def ready(self) -> bool:
        return self._classifier is not None

    @property
    def classifier(self) -> SpamClassifier:
        return self.ensure_loaded()

    def load(self) -> SpamClassifier:
        """(Re)load the model from disk. Raises FileNotFoundError if absent."""
        with self._lock:
            classifier = SpamClassifier.load(Path(self.settings.model_path))
            self._classifier = classifier
            self._loaded_at = datetime.now(timezone.utc)
            return classifier

    def ensure_loaded(self) -> SpamClassifier:
        """Load on first use; optionally train when the artifact is missing."""
        if self._classifier is not None:
            return self._classifier

        with self._lock:
            if self._classifier is not None:
                return self._classifier
            if not Path(self.settings.model_path).exists():
                if not self.settings.autotrain:
                    raise ModelNotReady(
                        f"No model at {self.settings.model_path}. "
                        "Run 'python main.py train' or set SPAM_AUTOTRAIN=1."
                    )
                from src.train import train  # imported lazily: training is slow

                train(n_jobs=1, verbose=False)
            return self.load()

    def reload(self) -> SpamClassifier:
        return self.load()

    # ------------------------------------------------------------- inference
    def predict(
        self,
        text: str,
        threshold: float | None = None,
        explain: bool = False,
    ) -> dict:
        classifier = self.ensure_loaded()
        cutoff = self.settings.threshold if threshold is None else threshold

        probability = float(classifier.spam_probability([text])[0])
        label = "spam" if probability >= cutoff else "ham"
        return {
            "label": label,
            "spam_probability": round(probability, 6),
            "confidence": round(max(probability, 1.0 - probability), 6),
            "threshold": cutoff,
            "explanation": (
                classifier.explain(text, top_k=self.settings.explain_top_k)
                if explain
                else []
            ),
        }

    def predict_many(
        self,
        messages: list[str],
        threshold: float | None = None,
        explain: bool = False,
    ) -> list[dict]:
        classifier = self.ensure_loaded()
        cutoff = self.settings.threshold if threshold is None else threshold

        probabilities = classifier.spam_probability(messages)
        results = []
        for message, probability in zip(messages, probabilities, strict=True):
            probability = float(probability)
            results.append(
                {
                    "message": message,
                    "label": SPAM if probability >= cutoff else "ham",
                    "spam_probability": round(probability, 6),
                    "confidence": round(max(probability, 1.0 - probability), 6),
                    "explanation": (
                        classifier.explain(message, top_k=self.settings.explain_top_k)
                        if explain
                        else []
                    ),
                }
            )
        return results

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        classifier = self.ensure_loaded()
        metadata = classifier.metadata or {}
        test_metrics = metadata.get("test", {})
        return {
            "name": metadata.get("model", "unknown"),
            "version": classifier.version,
            "trained_at": metadata.get("trained_at"),
            "sklearn_version": metadata.get("sklearn_version"),
            "dataset": metadata.get("dataset", {}),
            "metrics": {
                key: test_metrics.get(key)
                for key in ("accuracy", "macro_f1", "roc_auc")
                if key in test_metrics
            },
            "spam_metrics": test_metrics.get("spam", {}),
            "best_params": metadata.get("best_params", {}),
            "cv_f1_spam": metadata.get("cv_f1_spam"),
            "leaderboard": metadata.get("leaderboard", []),
            "threshold": self.settings.threshold,
            "loaded_at": (
                self._loaded_at.isoformat(timespec="seconds")
                if self._loaded_at
                else datetime.now(timezone.utc).isoformat(timespec="seconds")
            ),
            "explainable": True,
        }
