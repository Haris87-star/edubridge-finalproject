"""HTTP client for the SMS Spam Detector API.

Used by the Streamlit dashboard, but handy from any Python code::

    from src.client import SpamAPIClient

    with SpamAPIClient() as api:
        api.predict("You won a free prize, call now!")
"""

from __future__ import annotations

import os

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class APIError(RuntimeError):
    """The API could not be reached or returned an error."""


class SpamAPIClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0):
        base_url = base_url or os.getenv("SPAM_API_URL", DEFAULT_API_URL)
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SpamAPIClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ----------------------------------------------------------------- calls
    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise APIError(f"Cannot reach the API at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise APIError(f"API error {response.status_code}: {detail}")

        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def ready(self) -> dict:
        return self._request("GET", "/ready")

    def model_info(self) -> dict:
        return self._request("GET", "/model/info")

    def stats(self) -> dict:
        return self._request("GET", "/stats")

    def predict(
        self,
        text: str,
        explain: bool = False,
        threshold: float | None = None,
    ) -> dict:
        payload: dict = {"text": text, "explain": explain}
        if threshold is not None:
            payload["threshold"] = threshold
        return self._request("POST", "/predict", json=payload)

    def predict_batch(
        self,
        messages: list[str],
        explain: bool = False,
        threshold: float | None = None,
    ) -> dict:
        payload: dict = {"messages": messages, "explain": explain}
        if threshold is not None:
            payload["threshold"] = threshold
        return self._request("POST", "/predict/batch", json=payload)

    def feedback(
        self,
        text: str,
        actual: str,
        predicted: str | None = None,
        notes: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/feedback",
            json={"text": text, "actual": actual, "predicted": predicted, "notes": notes},
        )
