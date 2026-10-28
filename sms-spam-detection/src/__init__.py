"""SMS spam detection package.

Modules
-------
config        paths and constants
preprocessing text normalisation (shared by training and inference)
dataset       dataset download / parsing / caching
model         pipelines, persistence, :class:`SpamClassifier`
train         cross-validated training
evaluate      metrics and reporting

Typical use::

    from src.model import SpamClassifier
    from src.preprocessing import clean_text
    from src.train import train

    train()
    clf = SpamClassifier.load()
    clf.predict_one("You have won a free prize, call now!")
"""

__version__ = "1.0.0"
