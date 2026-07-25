# src/health.py
"""Health check and readiness endpoints."""

from __future__ import annotations

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter

from src.config.settings import get_settings

logger = logging.getLogger("socraites.health")
router = APIRouter(tags=["health"])

# Track application start time
_start_time = time.time()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Liveness probe — indicates the app process is running."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "service": "socraites",
    }


@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """Readiness probe — checks if dependencies are reachable."""
    settings = get_settings()
    checks: Dict[str, str] = {}

    # Check Elasticsearch connectivity
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.es_url}/_cluster/health")
            if resp.status_code == 200:
                checks["elasticsearch"] = "ok"
            else:
                checks["elasticsearch"] = f"unhealthy (status={resp.status_code})"
    except Exception as e:
        checks["elasticsearch"] = f"unreachable ({type(e).__name__})"

    # Overall readiness
    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "environment": settings.app_env,
    }
