"""Model definitions, persistence, and the inference wrapper."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import MODEL_FILE, SPAM
from .preprocessing import clean_text

__all__ = [
    "make_vectorizer",
    "candidate_models",
    "spam_f1_scorer",
    "SpamClassifier",
    "save_model",
]


def make_vectorizer() -> TfidfVectorizer:
    """TF-IDF over cleaned word uni- and bi-grams.

    ``preprocessor=clean_text`` means the pipeline accepts raw SMS text, so
    there is no way to forget the cleaning step at prediction time.
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    """Candidate pipelines and the hyper-parameters to search for each."""
    return {
        "naive_bayes": (
            Pipeline([("tfidf", make_vectorizer()), ("clf", MultinomialNB())]),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
        "logistic_regression": (
            Pipeline(
                [
                    ("tfidf", make_vectorizer()),
                    (
                        "clf",
                        LogisticRegression(
                            solver="liblinear",
                            max_iter=1000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {"clf__C": [1.0, 10.0, 100.0]},
        ),
        "linear_svm": (
            Pipeline(
                [
                    ("tfidf", make_vectorizer()),
                    (
                        "clf",
                        # LinearSVC has no predict_proba; calibration adds it so
                        # the app can show a confidence score.
                        CalibratedClassifierCV(
                            LinearSVC(class_weight="balanced"), cv=3
                        ),
                    ),
                ]
            ),
            {"clf__estimator__C": [0.5, 1.0]},
        ),
    }


def spam_f1_scorer():
    """F1 for the spam class - the metric that matters most here."""
    return make_scorer(f1_score, pos_label=SPAM)


def save_model(classifier: SpamClassifier, path: Path = MODEL_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": classifier.pipeline,
            "metadata": classifier.metadata,
            "version": classifier.version,
        },
        path,
    )
    return path


class SpamClassifier:
    """Thin wrapper around the trained scikit-learn pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: dict | None = None,
        version: str | None = None,
    ):
        self.pipeline = pipeline
        self.metadata = metadata or {}
        self.version = version or _metadata_digest(self.metadata)
        self.classes_ = [str(c) for c in pipeline.classes_]

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> SpamClassifier:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: python main.py train"
            )
        artifact = joblib.load(path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            return cls(
                artifact["pipeline"],
                artifact.get("metadata"),
                artifact.get("version") or _file_digest(path),
            )
        return cls(artifact, version=_file_digest(path))

    # ------------------------------------------------------------ inference
    def predict(self, texts: str | Iterable[str]) -> list[str]:
        return [str(label) for label in self.pipeline.predict(_as_list(texts))]

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    def predict_proba(self, texts: str | Iterable[str]) -> np.ndarray:
        """Probability per class, columns ordered as :attr:`classes_`."""
        return self.pipeline.predict_proba(_as_list(texts))

    def spam_probability(self, texts: str | Iterable[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        return proba[:, self.classes_.index(SPAM)]

    def is_spam(self, text: str, threshold: float = 0.5) -> bool:
        return bool(self.spam_probability([text])[0] >= threshold)

    # ------------------------------------------------------- explainability
    def explain(self, text: str, top_k: int = 8) -> list[dict]:
        """Tokens that pushed this specific message towards spam or ham.

        Works for linear models and Naive Bayes. Returns an empty list for
        models that cannot be interpreted this way.
        """
        weights = _classifier_weights(self.pipeline.named_steps["clf"])
        if weights is None:
            return []

        vectorizer = self.pipeline.named_steps["tfidf"]
        row = vectorizer.transform([text])
        names = vectorizer.get_feature_names_out()
        indices = row.indices
        if len(indices) == 0:
            return []

        contributions = row.data * weights[indices]
        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        return [
            {
                "token": str(names[indices[i]]),
                "weight": round(float(contributions[i]), 4),
            }
            for i in order
        ]

    # ------------------------------------------------------------------ info
    @property
    def name(self) -> str:
        return self.metadata.get("model", "unknown")

    @property
    def metrics(self) -> dict:
        return self.metadata.get("test", {})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SpamClassifier(model={self.name!r}, classes={self.classes_})"


def _as_list(texts: str | Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(text) for text in texts]


def _file_digest(path: Path, length: int = 12) -> str:
    """Short content hash used as the model version."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _metadata_digest(metadata: dict, length: int = 12) -> str:
    """Stable version for in-memory models that have not been written to disk."""
    payload = json.dumps(metadata or {}, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _classifier_weights(clf) -> np.ndarray | None:
    """Signed per-feature weights for the spam class."""
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_)[0]

    if hasattr(clf, "feature_log_prob_"):  # MultinomialNB
        log_prob = np.asarray(clf.feature_log_prob_)
        return log_prob[1] - log_prob[0]

    for attribute in ("estimator", "base_estimator"):
        inner = getattr(clf, attribute, None)
        if inner is not None:
            return _classifier_weights(inner)

    # CalibratedClassifierCV stores one fitted estimator per fold.
    calibrated = getattr(clf, "calibrated_classifiers_", None)
    if calibrated:
        return _classifier_weights(calibrated[0].estimator)

    return None
