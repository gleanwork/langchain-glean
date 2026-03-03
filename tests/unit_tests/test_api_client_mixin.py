import os
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class _MixinConcrete(GleanAPIClientMixin, BaseModel):
    """Concrete class for testing the mixin (cannot instantiate mixin alone)."""

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

    def test_resolves_from_env_vars(self) -> None:
        """Test that the mixin resolves values from environment variables."""
        mixin = _MixinConcrete()
        assert mixin.instance == "test-instance"
        assert mixin.api_token == "test-token"
        assert mixin.act_as == "user@example.com"

    def test_resolves_from_constructor_args(self) -> None:
        """Test that constructor args take precedence over env vars."""
        mixin = _MixinConcrete(
            instance="custom-instance",
            api_token="custom-token",
            act_as="other@example.com",
        )
        assert mixin.instance == "custom-instance"
        assert mixin.api_token == "custom-token"
        assert mixin.act_as == "other@example.com"

    def test_missing_instance_raises(self) -> None:
        """Test that missing GLEAN_INSTANCE raises ValueError."""
        del os.environ["GLEAN_INSTANCE"]
        with pytest.raises(ValueError):
            _MixinConcrete()

    def test_missing_api_token_raises(self) -> None:
        """Test that missing GLEAN_API_TOKEN raises ValueError."""
        del os.environ["GLEAN_API_TOKEN"]
        with pytest.raises(ValueError):
            _MixinConcrete()

    def test_act_as_defaults_to_empty(self) -> None:
        """Test that act_as defaults to empty string when not set."""
        del os.environ["GLEAN_ACT_AS"]
        mixin = _MixinConcrete()
        assert mixin.act_as == ""

    def test_http_headers_with_act_as(self) -> None:
        """Test _http_headers returns ActAs header when act_as is set."""
        mixin = _MixinConcrete()
        headers = mixin._http_headers()
        assert headers == {"X-Glean-ActAs": "user@example.com"}

    def test_http_headers_without_act_as(self) -> None:
        """Test _http_headers returns None when act_as is not set."""
        del os.environ["GLEAN_ACT_AS"]
        mixin = _MixinConcrete()
        headers = mixin._http_headers()
        assert headers is None

    def test_http_headers_with_empty_act_as(self) -> None:
        """Test _http_headers returns None when act_as is empty string."""
        os.environ["GLEAN_ACT_AS"] = ""
        mixin = _MixinConcrete()
        headers = mixin._http_headers()
        assert headers is None
