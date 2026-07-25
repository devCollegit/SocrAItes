# tests/conftest.py
"""Shared pytest fixtures for SocrAItes tests."""

import os
import pytest
import tempfile

# Set test environment before importing application code
os.environ["APP_ENV"] = "testing"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ES_URL"] = "http://localhost:9200"


@pytest.fixture(scope="session")
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    os.environ["SOCRAITES_DB_PATH"] = db_path
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def test_client(temp_db_path):
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api import app
    with TestClient(app) as client:
        yield client
