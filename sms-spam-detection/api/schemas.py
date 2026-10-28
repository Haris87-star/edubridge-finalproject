"""Request and response models (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Label = Literal["ham", "spam"]


class ModelRef(BaseModel):
    name: str
    version: str


class TokenContribution(BaseModel):
    token: str = Field(description="Vocabulary token that influenced the decision.")
    weight: float = Field(description="Signed contribution: positive pushes towards spam.")


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Congratulations! You won a £1000 prize. Call 0800 123 4567 now!",
                    "explain": True,
                },
                {"text": "Hey, are we still meeting for lunch tomorrow?"},
            ]
        }
    )

    text: str = Field(min_length=1, max_length=5000, description="Raw SMS text.")
    explain: bool = Field(default=False, description="Return top contributing tokens.")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the server's spam threshold for this request.",
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class PredictionResponse(BaseModel):
    label: Label
    spam_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the chosen label.")
    threshold: float
    explanation: list[TokenContribution] = Field(default_factory=list)
    model: ModelRef
    latency_ms: float
    request_id: str


class BatchPredictRequest(BaseModel):
    messages: list[str] = Field(min_length=1, description="Messages to classify.")
    explain: bool = False
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def _clean_messages(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("messages must not contain blank entries")
        if any(len(value) > 5000 for value in cleaned):
            raise ValueError("each message must be at most 5000 characters")
        return cleaned


class BatchItem(BaseModel):
    message: str
    label: Label
    spam_probability: float
    confidence: float
    explanation: list[TokenContribution] = Field(default_factory=list)


class BatchSummary(BaseModel):
    total: int
    spam: int
    ham: int
    spam_ratio: float
    threshold: float


class BatchPredictionResponse(BaseModel):
    results: list[BatchItem]
    summary: BatchSummary
    model: ModelRef
    latency_ms: float


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    trained_at: str | None = None
    sklearn_version: str | None = None
    dataset: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    spam_metrics: dict = Field(default_factory=dict)
    best_params: dict = Field(default_factory=dict)
    cv_f1_spam: float | None = None
    leaderboard: list[dict] = Field(default_factory=list)
    threshold: float
    loaded_at: str
    explainable: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok"]
    uptime_seconds: float
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    model: str
    version: str


class FeedbackRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    actual: Label
    predicted: Label | None = None
    notes: str | None = Field(default=None, max_length=1000)
    source: str | None = Field(default=None, max_length=100)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class FeedbackResponse(BaseModel):
    stored: bool
    path: str


class ReloadResponse(BaseModel):
    reloaded: bool
    version: str
