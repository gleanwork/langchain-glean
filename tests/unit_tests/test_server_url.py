"""Tests for server_url support in GleanAPIClientMixin."""

import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_glean._api_client_mixin import GleanAPIClientMixin
from langchain_glean.chat_models.chat import ChatGlean


class TestServerUrl:
    """Test server_url field and _build_glean_client()."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Clear env vars before each test
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS", "GLEAN_SERVER_URL"]:
            os.environ.pop(var, None)

        os.environ["GLEAN_API_TOKEN"] = "test-token"

        yield

        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS", "GLEAN_SERVER_URL"]:
            os.environ.pop(var, None)

    def test_server_url_param(self):
        """Test that server_url can be passed directly."""
        chat = ChatGlean(server_url="https://acme-be.glean.com")
        assert chat.server_url == "https://acme-be.glean.com"

    def test_server_url_env_var(self):
        """Test that GLEAN_SERVER_URL env var is picked up."""
        os.environ["GLEAN_SERVER_URL"] = "https://env-be.glean.com"
        chat = ChatGlean()
        assert chat.server_url == "https://env-be.glean.com"

    def test_server_url_takes_precedence_over_instance(self):
        """Test that server_url takes precedence when both are set."""
        os.environ["GLEAN_INSTANCE"] = "should-not-be-used"
        chat = ChatGlean(server_url="https://acme-be.glean.com")
        assert chat.server_url == "https://acme-be.glean.com"

    def test_server_url_env_takes_precedence_over_instance_env(self):
        """Test that GLEAN_SERVER_URL env var takes precedence over GLEAN_INSTANCE env var."""
        os.environ["GLEAN_SERVER_URL"] = "https://env-be.glean.com"
        os.environ["GLEAN_INSTANCE"] = "should-not-be-used"
        chat = ChatGlean()
        assert chat.server_url == "https://env-be.glean.com"

    def test_instance_fallback_still_works(self):
        """Test that instance param still works when server_url is not set."""
        os.environ["GLEAN_INSTANCE"] = "acme"
        chat = ChatGlean()
        assert chat.instance == "acme"
        assert not chat.server_url

    def test_instance_param_still_works(self):
        """Test that instance can be passed directly."""
        chat = ChatGlean(instance="acme")
        assert chat.instance == "acme"
        assert not chat.server_url

    def test_build_glean_client_with_server_url(self):
        """Test _build_glean_client uses server_url when set."""
        chat = ChatGlean(server_url="https://acme-be.glean.com")
        with patch("langchain_glean._api_client_mixin.Glean") as mock_glean:
            chat._build_glean_client()
            mock_glean.assert_called_once_with(
                server_url="https://acme-be.glean.com",
                api_token="test-token",
            )

    def test_build_glean_client_with_instance(self):
        """Test _build_glean_client uses instance when server_url is not set."""
        chat = ChatGlean(instance="acme")
        with patch("langchain_glean._api_client_mixin.Glean") as mock_glean:
            chat._build_glean_client()
            mock_glean.assert_called_once_with(
                instance="acme",
                api_token="test-token",
            )

    def test_missing_both_server_url_and_instance_raises(self):
        """Test that omitting both server_url and instance raises ValueError."""
        with pytest.raises(ValueError):
            ChatGlean()
