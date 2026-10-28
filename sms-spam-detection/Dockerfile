# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder ----
# Installs dependencies and trains the model so the runtime image is
# self-contained and reproducible (fixed seed, dataset baked in).
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/install/lib/python3.12/site-packages

WORKDIR /build

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY src ./src
COPY main.py ./
RUN python main.py train --n-jobs 1

# ---------------------------------------------------------------- runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/install/lib/python3.12/site-packages \
    SPAM_MODEL_PATH=/app/models/sms_spam_model.joblib \
    SPAM_FEEDBACK_PATH=/app/feedback/feedback.jsonl \
    SPAM_LOG_LEVEL=INFO

WORKDIR /app

COPY --from=builder /install /install
COPY --from=builder /build/models ./models
COPY --from=builder /build/data ./data
COPY api ./api
COPY src ./src

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/feedback \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
