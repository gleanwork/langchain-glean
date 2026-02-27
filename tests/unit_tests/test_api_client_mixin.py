import os
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class _MixinTestModel(GleanAPIClientMixin, BaseModel):
    """Minimal concrete model for testing GleanAPIClientMixin."""

    pass


class TestGleanAPIClientMixin:
    """Test the GleanAPIClientMixin class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        # Ensure env vars are clean before each test
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

        yield

        # Clean up after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    # ===== ENVIRONMENT VARIABLE RESOLUTION =====

    def test_resolve_from_env_vars(self) -> None:
        """Test that fields are resolved from environment variables."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "user@example.com"

        model = _MixinTestModel()

        assert model.instance == "test-instance"
        assert model.api_token == "test-token"
        assert model.act_as == "user@example.com"

    def test_resolve_from_constructor_args(self) -> None:
        """Test that constructor arguments take precedence over env vars."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"

        model = _MixinTestModel(instance="arg-instance", api_token="arg-token")

        assert model.instance == "arg-instance"
        assert model.api_token == "arg-token"

    def test_constructor_args_override_env_vars(self) -> None:
        """Test that constructor args override all env vars including act_as."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"
        os.environ["GLEAN_ACT_AS"] = "env-user@example.com"

        model = _MixinTestModel(
            instance="arg-instance",
            api_token="arg-token",
            act_as="arg-user@example.com",
        )

        assert model.instance == "arg-instance"
        assert model.api_token == "arg-token"
        assert model.act_as == "arg-user@example.com"

    def test_missing_instance_raises_error(self) -> None:
        """Test that missing GLEAN_INSTANCE raises ValueError."""
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        with pytest.raises(ValueError):
            _MixinTestModel()

    def test_missing_api_token_raises_error(self) -> None:
        """Test that missing GLEAN_API_TOKEN raises ValueError."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"

        with pytest.raises(ValueError):
            _MixinTestModel()

    def test_missing_both_required_raises_error(self) -> None:
        """Test that missing both required fields raises ValueError."""
        with pytest.raises(ValueError):
            _MixinTestModel()

    def test_act_as_defaults_to_empty_string(self) -> None:
        """Test that act_as defaults to empty string when not provided."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        model = _MixinTestModel()

        assert model.act_as == ""

    # ===== HTTP HEADERS =====

    def test_http_headers_with_act_as(self) -> None:
        """Test _http_headers returns ActAs header when act_as is set."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "user@example.com"

        model = _MixinTestModel()
        headers = model._http_headers()

        assert headers == {"X-Glean-ActAs": "user@example.com"}

    def test_http_headers_without_act_as(self) -> None:
        """Test _http_headers returns None when act_as is not set."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"

        model = _MixinTestModel()
        headers = model._http_headers()

        assert headers is None

    def test_http_headers_with_empty_act_as(self) -> None:
        """Test _http_headers returns None when act_as is empty string."""
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = ""

        model = _MixinTestModel()
        headers = model._http_headers()

        assert headers is None
