"""Tests for authentication helper functions."""
import pytest
from fastapi import HTTPException

from main import _extract_bearer_token


class TestExtractBearerToken:
    """Tests for _extract_bearer_token function."""

    def test_extract_bearer_token_valid(self):
        """Extracts token from valid Bearer header."""
        token = _extract_bearer_token("Bearer my-test-token")
        assert token == "my-test-token"

    def test_extract_bearer_token_valid_case_insensitive(self):
        """Extracts token with case-insensitive Bearer prefix."""
        token = _extract_bearer_token("bearer my-test-token")
        assert token == "my-test-token"

    def test_extract_bearer_token_missing(self):
        """Raises 401 when authorization header is None."""
        with pytest.raises(HTTPException) as exc_info:
            _extract_bearer_token(None)
        assert exc_info.value.status_code == 401
        assert "missing" in exc_info.value.detail.lower()

    def test_extract_bearer_token_no_bearer_prefix(self):
        """Raises 401 when Bearer prefix is missing."""
        with pytest.raises(HTTPException) as exc_info:
            _extract_bearer_token("Basic my-token")
        assert exc_info.value.status_code == 401

    def test_extract_bearer_token_empty(self):
        """Raises 401 for empty authorization header."""
        with pytest.raises(HTTPException) as exc_info:
            _extract_bearer_token("")
        assert exc_info.value.status_code == 401


class TestDecodeClerkToken:
    """Tests for _decode_clerk_token function."""

    def test_decode_clerk_token_missing(self):
        """Raises 401 when token is empty."""
        from main import _decode_clerk_token

        with pytest.raises(HTTPException) as exc_info:
            _decode_clerk_token("")
        assert exc_info.value.status_code == 401
        assert "missing" in exc_info.value.detail.lower()

    def test_decode_clerk_token_invalid(self):
        """Raises 401 for invalid JWT format."""
        from main import _decode_clerk_token

        with pytest.raises(HTTPException) as exc_info:
            _decode_clerk_token("not-a-valid-jwt")
        assert exc_info.value.status_code == 401

    def test_decode_clerk_token_valid_unsigned(self):
        """Decodes valid unsigned JWT in dev mode (no JWKS configured)."""
        import jwt
        from main import _decode_clerk_token

        # Create a simple JWT with sub claim
        token = jwt.encode({"sub": "user_123"}, "secret", algorithm="HS256")
        # In dev mode (no JWKS), it decodes without verification
        payload = _decode_clerk_token(token)
        assert payload["sub"] == "user_123"


class TestVerifyApiKey:
    """Tests for verify_api_key dependency."""

    @pytest.mark.asyncio
    async def test_verify_api_key_missing(self, test_client, test_org):
        """Returns 401 when X-API-Key header is missing."""
        # Make a request that requires API key but don't provide one
        response = await test_client.post(
            "/api/emails",
            json={
                "sender": "test@example.com",
                "recipient": "to@example.com",
                "subject": "Test",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_api_key_invalid(self, test_client):
        """Returns 401 for invalid API key."""
        response = await test_client.post(
            "/api/emails",
            json={
                "sender": "test@example.com",
                "recipient": "to@example.com",
                "subject": "Test",
            },
            headers={"x-api-key": "invalid-key-12345"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_api_key_valid(self, test_client, test_org):
        """Returns 201 with valid API key."""
        response = await test_client.post(
            "/api/emails",
            json={
                "sender": "test@example.com",
                "recipient": "to@example.com",
                "subject": "Test",
            },
            headers={"x-api-key": test_org["api_key"]},
        )
        assert response.status_code == 201


class TestRequireAdmin:
    """Tests for require_admin dependency."""

    @pytest.mark.asyncio
    async def test_require_admin_success(self, test_client, test_admin_user):
        """Allows admin user through."""
        from tests.conftest import create_mock_jwt

        token = create_mock_jwt(test_admin_user["clerk_id"])
        response = await test_client.get(
            "/api/organizations",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_require_admin_forbidden(self, test_client, test_member_user):
        """Raises 403 for non-admin users."""
        from tests.conftest import create_mock_jwt

        token = create_mock_jwt(test_member_user["clerk_id"])
        response = await test_client.get(
            "/api/organizations",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


class TestResolveIngestContext:
    """Tests for resolve_ingest_context dependency."""

    @pytest.mark.asyncio
    async def test_resolve_ingest_context_api_key(self, test_client, test_org):
        """Resolves organisation from API key."""
        response = await test_client.post(
            "/api/emails",
            json={
                "sender": "test@example.com",
                "recipient": "to@example.com",
                "subject": "Test via API Key",
            },
            headers={"x-api-key": test_org["api_key"]},
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_resolve_ingest_context_bearer(self, test_client, test_admin_user):
        """Resolves organisation from Bearer token."""
        from tests.conftest import create_mock_jwt

        token = create_mock_jwt(test_admin_user["clerk_id"])
        response = await test_client.post(
            "/api/emails",
            json={
                "sender": "test@example.com",
                "recipient": "to@example.com",
                "subject": "Test via Bearer",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_resolve_ingest_context_missing(self, test_client):
        """Returns 401 when no auth provided."""
        response = await test_client.post(
            "/api/emails",
            json={
                "sender": "test@example.com",
                "recipient": "to@example.com",
                "subject": "Test",
            },
        )
        assert response.status_code == 401
