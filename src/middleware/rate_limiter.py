# src/middleware/rate_limiter.py
"""Rate limiting middleware using in-memory storage (Redis-backed in production)."""

from __future__ import annotations

import time
import logging
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.config.settings import get_settings

logger = logging.getLogger("socraites.ratelimit")


class InMemoryRateLimiter:
    """Simple sliding-window rate limiter using in-memory storage.

    In production with multiple workers, replace with Redis-backed implementation.
    """

    def __init__(self):
        # {client_key: [(timestamp, count), ...]}
        self._requests: Dict[str, list] = defaultdict(list)

    def _cleanup(self, key: str, window_seconds: int):
        """Remove expired entries."""
        now = time.time()
        cutoff = now - window_seconds
        self._requests[key] = [
            (ts, count) for ts, count in self._requests[key] if ts > cutoff
        ]

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        self._cleanup(key, window_seconds)

        total = sum(count for _, count in self._requests[key])
        if total >= max_requests:
            return False, 0

        self._requests[key].append((time.time(), 1))
        remaining = max_requests - total - 1
        return True, remaining


# Global rate limiter instance
_limiter = InMemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        settings = get_settings()

        # Skip rate limiting for health checks
        path = request.url.path
        if path in ("/health", "/readiness", "/metrics"):
            return await call_next(request)

        # Get client identifier (IP or user_id from JWT)
        client_ip = request.client.host if request.client else "unknown"
        client_key = f"ip:{client_ip}"

        # Determine rate limit based on endpoint
        if path == "/upload":
            max_requests = 10  # 10 uploads per minute
        elif path == "/chat":
            max_requests = settings.rate_limit_per_minute
        else:
            max_requests = settings.rate_limit_per_minute * 2  # More lenient for other endpoints

        allowed, remaining = _limiter.is_allowed(client_key, max_requests, window_seconds=60)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_key} on {path}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
