import os
from unittest.mock import patch

import pytest

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class TestGleanAPIClientMixin:
    """Test the GleanAPIClientMixin shared auth and client configuration.

    Uses GleanSearchRetriever as a concrete class that inherits the mixin,
    since the mixin is designed to be used via multiple inheritance with
    Pydantic models (BaseRetriever, BaseChatModel, BaseTool).
    """

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Clean environment before and after each test."""
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

        yield

        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def _create_retriever(self, **kwargs):
        """Helper to create a GleanSearchRetriever (concrete mixin user)."""
        from langchain_glean.retrievers.search import GleanSearchRetriever

        with patch("langchain_glean.retrievers.search.Glean"):
            return GleanSearchRetriever(**kwargs)

    def test_resolve_from_env_vars(self) -> None:
        """Test that instance and api_token are resolved from env vars."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"

        retriever = self._create_retriever()

        assert retriever.instance == "env-instance"
        assert retriever.api_token == "env-token"

    def test_resolve_from_constructor_args(self) -> None:
        """Test that constructor args take precedence over env vars."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"

        retriever = self._create_retriever(instance="arg-instance", api_token="arg-token")

        assert retriever.instance == "arg-instance"
        assert retriever.api_token == "arg-token"

    def test_missing_instance_raises(self) -> None:
        """Test that missing GLEAN_INSTANCE raises ValueError."""
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        with pytest.raises(ValueError):
            self._create_retriever()

    def test_missing_api_token_raises(self) -> None:
        """Test that missing GLEAN_API_TOKEN raises ValueError."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"

        with pytest.raises(ValueError):
            self._create_retriever()

    def test_act_as_from_env(self) -> None:
        """Test that act_as is resolved from GLEAN_ACT_AS env var."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "user@example.com"

        retriever = self._create_retriever()

        assert retriever.act_as == "user@example.com"

    def test_act_as_from_constructor(self) -> None:
        """Test that act_as can be provided via constructor."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        retriever = self._create_retriever(act_as="admin@example.com")

        assert retriever.act_as == "admin@example.com"

    def test_act_as_defaults_to_empty(self) -> None:
        """Test that act_as defaults to empty string when not set."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        retriever = self._create_retriever()

        assert retriever.act_as == ""

    def test_http_headers_with_act_as(self) -> None:
        """Test that _http_headers returns impersonation headers when act_as is set."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        retriever = self._create_retriever(act_as="user@example.com")
        headers = retriever._http_headers()

        assert headers == {"X-Glean-ActAs": "user@example.com"}

    def test_http_headers_without_act_as(self) -> None:
        """Test that _http_headers returns None when act_as is not set."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        retriever = self._create_retriever()
        headers = retriever._http_headers()

        assert headers is None

    def test_constructor_args_override_env(self) -> None:
        """Test that constructor args override environment variables for all fields."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"
        os.environ["GLEAN_ACT_AS"] = "env-user@example.com"

        retriever = self._create_retriever(
            instance="arg-instance",
            api_token="arg-token",
            act_as="arg-user@example.com",
        )

        assert retriever.instance == "arg-instance"
        assert retriever.api_token == "arg-token"
        assert retriever.act_as == "arg-user@example.com"

    def test_mixin_works_with_multiple_concrete_classes(self) -> None:
        """Test that the mixin works across different concrete classes."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        with patch("langchain_glean.retrievers.search.Glean"):
            from langchain_glean.retrievers.search import GleanSearchRetriever

            search = GleanSearchRetriever()

        with patch("langchain_glean.retrievers.people.Glean"):
            from langchain_glean.retrievers.people import GleanPeopleProfileRetriever

            people = GleanPeopleProfileRetriever()

        # Both should have resolved the same env vars
        assert search.instance == "test-instance"
        assert people.instance == "test-instance"
        assert search.api_token == "test-token"
        assert people.api_token == "test-token"
