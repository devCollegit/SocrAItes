# tests/test_auth.py
"""Tests for authentication endpoints."""

import pytest


class TestAuthRegister:
    """Test user registration."""

    def test_register_success(self, test_client):
        """Should register a new user and return tokens."""
        response = test_client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "securepassword123",
            "name": "Test User",
        })
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["email"] == "test@example.com"
        assert data["token_type"] == "bearer"

    def test_register_duplicate_email(self, test_client):
        """Should return 409 for duplicate email."""
        # First registration
        test_client.post("/auth/register", json={
            "email": "duplicate@example.com",
            "password": "password123",
        })
        # Second registration with same email
        response = test_client.post("/auth/register", json={
            "email": "duplicate@example.com",
            "password": "password456",
        })
        assert response.status_code == 409


class TestAuthLogin:
    """Test user login."""

    def test_login_success(self, test_client):
        """Should login with valid credentials."""
        # Register first
        test_client.post("/auth/register", json={
            "email": "login@example.com",
            "password": "mypassword",
        })
        # Login
        response = test_client.post("/auth/login", json={
            "email": "login@example.com",
            "password": "mypassword",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    def test_login_invalid_password(self, test_client):
        """Should return 401 for wrong password."""
        # Register first
        test_client.post("/auth/register", json={
            "email": "loginwrong@example.com",
            "password": "correctpassword",
        })
        # Login with wrong password
        response = test_client.post("/auth/login", json={
            "email": "loginwrong@example.com",
            "password": "wrongpassword",
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self, test_client):
        """Should return 401 for non-existent user."""
        response = test_client.post("/auth/login", json={
            "email": "nonexistent@example.com",
            "password": "password",
        })
        assert response.status_code == 401


class TestAuthRefresh:
    """Test token refresh."""

    def test_refresh_token_success(self, test_client):
        """Should issue new tokens with valid refresh token."""
        # Register to get tokens
        reg_response = test_client.post("/auth/register", json={
            "email": "refresh@example.com",
            "password": "password123",
        })
        refresh_token = reg_response.json()["refresh_token"]

        # Refresh
        response = test_client.post("/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    def test_refresh_token_invalid(self, test_client):
        """Should return 401 for invalid refresh token."""
        response = test_client.post("/auth/refresh", json={
            "refresh_token": "invalid-token",
        })
        assert response.status_code == 401
