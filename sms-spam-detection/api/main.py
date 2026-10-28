"""FastAPI application: REST inference, monitoring, admin and the web UI."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .logging_conf import configure_logging
from .metrics import MetricsRegistry
from .schemas import (
    BatchItem,
    BatchPredictionResponse,
    BatchPredictRequest,
    BatchSummary,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelRef,
    PredictionResponse,
    PredictRequest,
    ReadyResponse,
    ReloadResponse,
)
from .service import ModelNotReady, ModelService
from .settings import Settings, get_settings

logger = logging.getLogger("spam.api")

STATIC_DIR = Path(__file__).parent / "static"
_feedback_lock = threading.Lock()


def create_app(
    settings: Settings | None = None,
    service: ModelService | None = None,
    metrics: MetricsRegistry | None = None,
) -> FastAPI:
    """Application factory.

    Dependencies can be injected, which is how the tests avoid touching the
    real model file.
    """
    settings = settings or get_settings()
    service = service or ModelService(settings)
    metrics = metrics or MetricsRegistry()

    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            classifier = service.ensure_loaded()
            logger.info(
                "model loaded",
                extra={
                    "fields": {
                        "model": classifier.name,
                        "version": classifier.version,
                        "path": str(settings.model_path),
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001 - the API must still start
            logger.warning(
                "model unavailable at startup",
                extra={"fields": {"error": str(exc)}},
            )
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "SMS spam classification with TF-IDF + linear models. "
            "Interactive docs at /docs, web UI at /."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service
    app.state.metrics = metrics

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------- observability
    @app.middleware("http")
    async def instrument(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started
            metrics.record_request(request.method, request.url.path, 500, duration)
            metrics.record_error("unhandled")
            logger.exception(
                "unhandled error",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                    }
                },
            )
            raise

        duration = time.perf_counter() - started
        route = request.scope.get("route")
        # Use the route template ("/predict"), not the raw path, to keep
        # metric label cardinality bounded.
        path = getattr(route, "path", request.url.path)
        metrics.record_request(request.method, path, response.status_code, duration)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{duration * 1000:.2f}"

        logger.info(
            "request",
            extra={
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
            },
        )
        return response

    @app.exception_handler(ModelNotReady)
    async def model_not_ready_handler(_: Request, exc: ModelNotReady):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # ------------------------------------------------------------- security
    def require_admin(authorization: str | None = Header(default=None)) -> None:
        token = settings.admin_token
        if not token:
            raise HTTPException(
                status_code=403,
                detail="Admin endpoints are disabled: set SPAM_ADMIN_TOKEN to enable them.",
            )
        if not authorization or not secrets.compare_digest(
            authorization, f"Bearer {token}"
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")

    # -------------------------------------------------------------- helpers
    def model_ref() -> ModelRef:
        info = service.ensure_loaded()
        return ModelRef(name=info.name, version=info.version)

    # --------------------------------------------------------------- routes
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", uptime_seconds=round(metrics.uptime_seconds, 1), version=settings.version
        )

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    def ready() -> ReadyResponse:
        if not service.ready:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        reference = model_ref()
        return ReadyResponse(status="ready", model=reference.name, version=reference.version)

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(**service.info())

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(payload: PredictRequest, request: Request) -> PredictionResponse:
        started = time.perf_counter()
        result = service.predict(payload.text, payload.threshold, payload.explain)
        metrics.record_prediction(result["label"])
        return PredictionResponse(
            **result,
            model=model_ref(),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            request_id=request.state.request_id,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictionResponse:
        if len(payload.messages) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Batch too large: {len(payload.messages)} messages, "
                    f"maximum is {settings.max_batch_size}."
                ),
            )

        started = time.perf_counter()
        results = service.predict_many(payload.messages, payload.threshold, payload.explain)
        for item in results:
            metrics.record_prediction(item["label"])

        spam_count = sum(1 for item in results if item["label"] == "spam")
        total = len(results)
        threshold = (
            payload.threshold if payload.threshold is not None else settings.threshold
        )

        return BatchPredictionResponse(
            results=[BatchItem(**item) for item in results],
            summary=BatchSummary(
                total=total,
                spam=spam_count,
                ham=total - spam_count,
                spam_ratio=round(spam_count / total, 4),
                threshold=threshold,
            ),
            model=model_ref(),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201, tags=["mlops"])
    def feedback(payload: FeedbackRequest) -> FeedbackResponse:
        path = Path(settings.feedback_path)
        record = payload.model_dump()
        record["received_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with _feedback_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics.record_feedback()
        logger.info("feedback stored", extra={"fields": {"actual": payload.actual}})
        return FeedbackResponse(stored=True, path=str(path))

    @app.post(
        "/admin/reload",
        response_model=ReloadResponse,
        dependencies=[Depends(require_admin)],
        tags=["mlops"],
    )
    def reload_model() -> ReloadResponse:
        try:
            classifier = service.reload()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        logger.info("model reloaded", extra={"fields": {"version": classifier.version}})
        return ReloadResponse(reloaded=True, version=classifier.version)

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def prometheus_metrics() -> str:
        return metrics.render_prometheus()

    @app.get("/stats", tags=["ops"])
    def stats() -> dict:
        return metrics.snapshot()

    # ------------------------------------------------------------------- UI
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
