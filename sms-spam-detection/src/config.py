"""Project-wide paths and settings.

Every path is derived from the location of this file, so the project works no
matter which directory you run it from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_FILE = DATA_DIR / "sms_spam.csv"
MODEL_FILE = MODELS_DIR / "sms_spam_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

HAM = "ham"
SPAM = "spam"
LABELS = (HAM, SPAM)
