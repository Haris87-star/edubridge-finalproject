#!/usr/bin/env python
"""Concurrent load test for POST /predict (stdlib only).

Usage:
    python scripts/load_test.py --requests 500 --concurrency 25
    python scripts/load_test.py --url http://localhost:8000 --requests 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MESSAGES = [
    "Congratulations! You won a £1000 cash prize. Call 0800 123 4567 now!",
    "Hey, are we still meeting for lunch tomorrow at 1?",
    "URGENT: your account will be suspended. Verify your details now.",
    "Ok, see you at the station in 20 minutes.",
]


def one_request(base_url: str, text: str, timeout: float) -> tuple[bool, float]:
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/predict",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            ok = response.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        ok = False
    return ok, (time.perf_counter() - started) * 1000


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"Load testing {base}/predict")
    print(f"  requests={args.requests} concurrency={args.concurrency}\n")

    latencies: list[float] = []
    errors = 0
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, base, MESSAGES[index % len(MESSAGES)], args.timeout)
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            ok, latency = future.result()
            latencies.append(latency)
            errors += 0 if ok else 1

    elapsed = time.perf_counter() - started
    succeeded = len(latencies) - errors
    throughput = succeeded / elapsed if elapsed else 0.0

    print(f"  duration .......... {elapsed:.2f} s")
    print(f"  succeeded ......... {succeeded}/{len(latencies)}")
    print(f"  errors ............ {errors}")
    print(f"  throughput ........ {throughput:.1f} req/s")
    print(f"  latency mean ...... {statistics.fmean(latencies):.2f} ms")
    print(f"  latency p50 ....... {percentile(latencies, 0.50):.2f} ms")
    print(f"  latency p95 ....... {percentile(latencies, 0.95):.2f} ms")
    print(f"  latency p99 ....... {percentile(latencies, 0.99):.2f} ms")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
