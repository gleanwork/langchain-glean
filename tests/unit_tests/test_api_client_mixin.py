import os

import pytest
from pydantic import BaseModel

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class _MixinConsumer(GleanAPIClientMixin, BaseModel):
    """Minimal concrete class that uses the mixin, for testing purposes."""

    pass


class TestGleanAPIClientMixin:
    """Test the GleanAPIClientMixin class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "user@example.com"

        yield

        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_resolve_env_from_environment(self) -> None:
        """Test that _resolve_env reads from environment variables."""
        mixin = _MixinConsumer()

        assert mixin.instance == "test-instance"
        assert mixin.api_token == "test-token"
        assert mixin.act_as == "user@example.com"

    def test_resolve_env_from_constructor_args(self) -> None:
        """Test that constructor args override environment variables."""
        mixin = _MixinConsumer(
            instance="custom-instance",
            api_token="custom-token",
            act_as="other@example.com",
        )

        assert mixin.instance == "custom-instance"
        assert mixin.api_token == "custom-token"
        assert mixin.act_as == "other@example.com"

    def test_resolve_env_missing_instance(self) -> None:
        """Test that missing GLEAN_INSTANCE raises ValueError."""
        os.environ.pop("GLEAN_INSTANCE", None)

        with pytest.raises(ValueError):
            _MixinConsumer()

    def test_resolve_env_missing_api_token(self) -> None:
        """Test that missing GLEAN_API_TOKEN raises ValueError."""
        os.environ.pop("GLEAN_API_TOKEN", None)

        with pytest.raises(ValueError):
            _MixinConsumer()

    def test_resolve_env_missing_act_as_defaults_to_empty(self) -> None:
        """Test that missing GLEAN_ACT_AS defaults to empty string."""
        os.environ.pop("GLEAN_ACT_AS", None)

        mixin = _MixinConsumer()

        assert mixin.act_as == ""

    def test_http_headers_with_act_as(self) -> None:
        """Test _http_headers returns X-Glean-ActAs header when act_as is set."""
        mixin = _MixinConsumer()

        headers = mixin._http_headers()

        assert headers == {"X-Glean-ActAs": "user@example.com"}

    def test_http_headers_without_act_as(self) -> None:
        """Test _http_headers returns None when act_as is not set."""
        os.environ.pop("GLEAN_ACT_AS", None)

        mixin = _MixinConsumer()

        headers = mixin._http_headers()

        assert headers is None

    def test_http_headers_with_empty_act_as(self) -> None:
        """Test _http_headers returns None when act_as is empty string."""
        os.environ.pop("GLEAN_ACT_AS", None)

        mixin = _MixinConsumer(act_as="")

        headers = mixin._http_headers()

        assert headers is None

    def test_constructor_args_take_precedence_over_env(self) -> None:
        """Test that explicit constructor args take precedence even when env vars exist."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"
        os.environ["GLEAN_ACT_AS"] = "env@example.com"

        mixin = _MixinConsumer(
            instance="arg-instance",
            api_token="arg-token",
            act_as="arg@example.com",
        )

        assert mixin.instance == "arg-instance"
        assert mixin.api_token == "arg-token"
        assert mixin.act_as == "arg@example.com"
