# tests/test_health.py
"""Tests for health check endpoints."""

import pytest


def test_health_endpoint(test_client):
    """Health endpoint should return 200 with status healthy."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert data["service"] == "socraites"


def test_readiness_endpoint(test_client):
    """Readiness endpoint should return 200 (may be degraded without ES)."""
    response = test_client.get("/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ready", "degraded")
    assert "checks" in data
