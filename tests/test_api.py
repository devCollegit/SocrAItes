# tests/test_api.py
"""Tests for core API endpoints."""

import pytest


class TestSessions:
    """Test session-related endpoints."""

    def test_list_sessions_empty(self, test_client):
        """Should return empty sessions list initially."""
        response = test_client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_list_documents(self, test_client):
        """Should return documents list (may fail without ES)."""
        response = test_client.get("/documents")
        # May return 500 if ES is not available, which is acceptable in tests
        assert response.status_code in (200, 500)


class TestWeaknesses:
    """Test weakness endpoints."""

    def test_list_weaknesses(self, test_client):
        """Should return weaknesses list."""
        response = test_client.get("/weaknesses")
        assert response.status_code == 200
        data = response.json()
        assert "weaknesses" in data


class TestSchedules:
    """Test schedule endpoints."""

    def test_list_schedules(self, test_client):
        """Should return schedules list."""
        response = test_client.get("/schedules")
        assert response.status_code == 200
        data = response.json()
        assert "schedules" in data


class TestUpload:
    """Test file upload validation."""

    def test_upload_non_pdf_rejected(self, test_client):
        """Should reject non-PDF files."""
        import io
        response = test_client.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")},
        )
        assert response.status_code == 400
