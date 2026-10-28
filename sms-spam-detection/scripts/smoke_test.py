#!/usr/bin/env python
"""Black-box smoke test against a running SMS Spam Detector API.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://localhost:8000 --wait 30

Exits with status 1 if any check fails, so it can gate a deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SPAM_TEXT = "Congratulations! You won a £1000 cash prize. Call 0800 123 4567 now!"
HAM_TEXT = "Hey, are we still meeting for lunch tomorrow at 1?"


def _request(url: str, method: str = "GET", payload: dict | None = None, timeout: float = 10.0):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def wait_until_ready(base_url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _ = _request(f"{base_url}/ready", timeout=3)
            if status == 200:
                return True
        except Exception:  # noqa: BLE001 - the server may not be up yet
            pass
        time.sleep(1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wait", type=float, default=30.0, help="seconds to wait for /ready")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"[{'PASS' if condition else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}")
        if not condition:
            failures.append(name)

    print(f"Smoke testing {base}")
    ready = wait_until_ready(base, args.wait)
    check("service becomes ready", ready, f"waited {args.wait:g}s")
    if not ready:
        print("\nService never became ready.", file=sys.stderr)
        return 1

    status, body = _request(f"{base}/health")
    health = json.loads(body) if body else {}
    check("GET /health", status == 200 and health.get("status") == "ok")

    status, body = _request(f"{base}/model/info")
    info = json.loads(body) if body else {}
    check(
        "GET /model/info",
        status == 200 and bool(info.get("version")) and info.get("name"),
        f"model={info.get('name')} v{info.get('version')}",
    )

    status, body = _request(f"{base}/predict", "POST", {"text": SPAM_TEXT, "explain": True})
    result = json.loads(body) if body else {}
    check(
        "POST /predict flags spam",
        status == 200 and result.get("label") == "spam",
        f"p={result.get('spam_probability')}",
    )
    check(
        "explanation returned",
        bool(result.get("explanation")),
        f"{len(result.get('explanation', []))} tokens",
    )

    status, body = _request(f"{base}/predict", "POST", {"text": HAM_TEXT})
    result = json.loads(body) if body else {}
    check(
        "POST /predict accepts ham",
        status == 200 and result.get("label") == "ham",
        f"p={result.get('spam_probability')}",
    )

    status, body = _request(
        f"{base}/predict/batch",
        "POST",
        {"messages": [SPAM_TEXT, HAM_TEXT, "Free entry! Text WIN to 87121"]},
    )
    batch = json.loads(body) if body else {}
    check(
        "POST /predict/batch",
        status == 200 and batch.get("summary", {}).get("total") == 3,
        f"summary={batch.get('summary')}",
    )

    status, body = _request(f"{base}/metrics")
    check(
        "GET /metrics is Prometheus text",
        status == 200 and "spam_api_predictions_total" in body,
    )

    status, body = _request(f"{base}/")
    check("web UI served", status == 200 and "SMS Spam Detector" in body)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
