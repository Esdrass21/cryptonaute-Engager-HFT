# data_ingestion/news_feeds/retry.py
"""
Rate-limit-aware HTTP client with exponential backoff + jitter.

Every external feed goes through this. No bare `requests.get` anywhere.

Design:
  * ONE session per client (connection pooling).
  * urllib3 Retry mounted on the session for automatic 429/5xx retry.
  * A token-bucket rate limiter in front so we don't even hit the limit.
  * Deterministic in replay mode: the caller passes `enabled=False` and no
    network call is ever made.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

log = logging.getLogger("data.http")


# --------------------------------------------------------------------------- #
# Token-bucket limiter                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class _Bucket:
    rate_per_sec: float
    capacity: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def take(self, n: float = 1.0) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
                self.last_refill = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                needed = n - self.tokens
                wait = needed / self.rate_per_sec if self.rate_per_sec > 0 else 0.05
            time.sleep(max(0.005, wait))


# --------------------------------------------------------------------------- #
# HTTP client                                                                  #
# --------------------------------------------------------------------------- #

class HttpClient:
    """
    Thin wrapper over requests.Session with retries, backoff, and rate limiting.

    Usage:
        http = HttpClient(rate_per_sec=2.0, user_agent="Cryptonaute/3.0")
        resp = http.get("https://api.example.com/data", params={...}, timeout=5.0)
    """

    RETRY_STATUS = (429, 500, 502, 503, 504)

    def __init__(
        self,
        *,
        rate_per_sec: float = 5.0,
        burst: float = 10.0,
        total_retries: int = 4,
        backoff_factor: float = 2.0,
        user_agent: str = "Cryptonaute/3.0 (research; contact: ops@example.invalid)",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.user_agent = user_agent
        self.bucket = _Bucket(rate_per_sec=rate_per_sec, capacity=burst)

        self.session = requests.Session()
        retry = Retry(
            total=total_retries,
            connect=total_retries,
            read=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=list(self.RETRY_STATUS),
            allowed_methods=frozenset(["GET", "POST", "HEAD"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, url: str, *, params: Optional[Dict[str, Any]] = None,
            timeout: float = 5.0, headers: Optional[Dict[str, str]] = None):
        if not self.enabled:
            raise RuntimeError("HttpClient disabled (replay mode)")
        self.bucket.take()
        h = {"User-Agent": self.user_agent, "Accept": "application/json, text/html"}
        if headers:
            h.update(headers)
        return self.session.get(url, params=params, timeout=timeout, headers=h)

    def post(self, url: str, *, json_body: Optional[Dict[str, Any]] = None,
             data: Optional[str] = None, timeout: float = 5.0,
             headers: Optional[Dict[str, str]] = None):
        if not self.enabled:
            raise RuntimeError("HttpClient disabled (replay mode)")
        self.bucket.take()
        h = {"User-Agent": self.user_agent, "Content-Type": "application/json"}
        if headers:
            h.update(headers)
        if data is not None:
            return self.session.post(url, data=data, timeout=timeout, headers=h)
        return self.session.post(url, json=json_body, timeout=timeout, headers=h)

    def close(self) -> None:
        self.session.close()
