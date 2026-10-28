"""Training: cross-validated model selection and evaluation."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .config import (
    CV_FOLDS,
    METRICS_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from .dataset import dataset_stats, ensure_dataset
from .evaluate import compute_metrics, format_report, save_metrics
from .model import SpamClassifier, candidate_models, save_model, spam_f1_scorer


def train(
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
    n_jobs: int = -1,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """Train every candidate, keep the best by cross-validated spam F1.

    Returns a dictionary with the fitted classifier, its metrics and the
    cross-validation leaderboard.
    """
    started = time.perf_counter()
    df = ensure_dataset(verbose=verbose)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    cross_validation = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=seed
    )
    scorer = spam_f1_scorer()

    leaderboard: list[dict] = []
    best_name, best_estimator, best_score = None, None, -1.0

    for name, (pipeline, grid) in candidate_models().items():
        if verbose:
            print(f"Training {name} (grid: {grid}) ...", flush=True)
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=scorer,
            cv=cross_validation,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)
        score = float(search.best_score_)
        leaderboard.append(
            {
                "model": name,
                "cv_f1_spam": round(score, 4),
                "best_params": search.best_params_,
            }
        )
        if verbose:
            print(f"  cv f1(spam) = {score:.4f}  {search.best_params_}", flush=True)
        if score > best_score:
            best_name, best_estimator, best_score = name, search.best_estimator_, score

    leaderboard.sort(key=lambda row: row["cv_f1_spam"], reverse=True)

    # The winning estimator was already refit on the full training set by
    # GridSearchCV(refit=True).
    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)[:, list(best_estimator.classes_).index("spam")]
    test_metrics = compute_metrics(y_test, y_pred, y_proba)

    metadata = {
        "model": best_name,
        "best_params": next(
            row["best_params"] for row in leaderboard if row["model"] == best_name
        ),
        "cv_f1_spam": best_score,
        "test": test_metrics,
        "leaderboard": leaderboard,
        "dataset": dataset_stats(df),
        "split": {"test_size": test_size, "random_state": seed, "cv_folds": cv_folds},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.perf_counter() - started, 1),
        "sklearn_version": sklearn.__version__,
    }

    classifier = SpamClassifier(best_estimator, metadata)

    if save:
        save_model(classifier, MODEL_FILE)
        save_metrics(metadata, METRICS_FILE)
        if verbose:
            print(f"Saved model  -> {MODEL_FILE}")
            print(f"Saved metrics -> {METRICS_FILE}")

    if verbose:
        print(format_report(test_metrics))
        print(f"\nSelected model: {best_name}  ({metadata['training_seconds']}s)")

    return {"classifier": classifier, "metrics": metadata, "leaderboard": leaderboard}
