import os
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class DummyMixinModel(GleanAPIClientMixin, BaseModel):
    """Minimal concrete class to test the mixin in isolation."""

    pass


class TestGleanAPIClientMixin:
    """Test the GleanAPIClientMixin class."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Remove Glean env vars before and after each test."""
        env_vars = ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]
        saved = {k: os.environ.pop(k, None) for k in env_vars}
        yield
        for k in env_vars:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    # ===== _resolve_env tests =====

    def test_resolve_from_constructor_args(self) -> None:
        """Fields supplied via kwargs take precedence."""
        model = DummyMixinModel(instance="acme", api_token="tok-123")
        assert model.instance == "acme"
        assert model.api_token == "tok-123"
        assert model.act_as == ""

    def test_resolve_from_env_vars(self) -> None:
        """Fields fall back to environment variables when not provided."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"

        model = DummyMixinModel()
        assert model.instance == "env-instance"
        assert model.api_token == "env-token"

    def test_constructor_overrides_env(self) -> None:
        """Constructor args take priority over env vars."""
        os.environ["GLEAN_INSTANCE"] = "env-instance"
        os.environ["GLEAN_API_TOKEN"] = "env-token"

        model = DummyMixinModel(instance="override", api_token="override-tok")
        assert model.instance == "override"
        assert model.api_token == "override-tok"

    def test_missing_instance_raises(self) -> None:
        """Missing GLEAN_INSTANCE should raise ValueError."""
        os.environ["GLEAN_API_TOKEN"] = "tok"
        with pytest.raises(ValueError):
            DummyMixinModel()

    def test_missing_api_token_raises(self) -> None:
        """Missing GLEAN_API_TOKEN should raise ValueError."""
        os.environ["GLEAN_INSTANCE"] = "acme"
        with pytest.raises(ValueError):
            DummyMixinModel()

    def test_missing_both_raises(self) -> None:
        """Missing both required fields should raise."""
        with pytest.raises(ValueError):
            DummyMixinModel()

    def test_act_as_from_env(self) -> None:
        """GLEAN_ACT_AS is optional and read from env."""
        os.environ["GLEAN_INSTANCE"] = "acme"
        os.environ["GLEAN_API_TOKEN"] = "tok"
        os.environ["GLEAN_ACT_AS"] = "user@example.com"

        model = DummyMixinModel()
        assert model.act_as == "user@example.com"

    def test_act_as_defaults_to_empty_string(self) -> None:
        """act_as defaults to empty string when not provided."""
        os.environ["GLEAN_INSTANCE"] = "acme"
        os.environ["GLEAN_API_TOKEN"] = "tok"

        model = DummyMixinModel()
        assert model.act_as == ""

    def test_act_as_from_constructor(self) -> None:
        """act_as can be set via constructor."""
        model = DummyMixinModel(
            instance="acme", api_token="tok", act_as="admin@example.com"
        )
        assert model.act_as == "admin@example.com"

    # ===== _http_headers tests =====

    def test_http_headers_with_act_as(self) -> None:
        """Returns X-Glean-ActAs header when act_as is set."""
        model = DummyMixinModel(
            instance="acme", api_token="tok", act_as="user@example.com"
        )
        headers = model._http_headers()
        assert headers == {"X-Glean-ActAs": "user@example.com"}

    def test_http_headers_without_act_as(self) -> None:
        """Returns None when act_as is empty."""
        model = DummyMixinModel(instance="acme", api_token="tok")
        headers = model._http_headers()
        assert headers is None

    def test_http_headers_with_empty_string_act_as(self) -> None:
        """Returns None when act_as is explicitly empty string."""
        model = DummyMixinModel(instance="acme", api_token="tok", act_as="")
        headers = model._http_headers()
        assert headers is None

    # ===== Edge cases =====

    def test_resolve_env_with_none_values_dict(self) -> None:
        """Validator handles values dict with None values gracefully."""
        os.environ["GLEAN_INSTANCE"] = "acme"
        os.environ["GLEAN_API_TOKEN"] = "tok"

        model = DummyMixinModel(instance=None, act_as=None)
        # instance=None should fall through to env var
        assert model.instance == "acme"
        assert model.api_token == "tok"

    def test_whitespace_act_as_is_truthy(self) -> None:
        """A non-empty act_as (even with spaces) produces headers."""
        model = DummyMixinModel(
            instance="acme", api_token="tok", act_as="  user@example.com  "
        )
        headers = model._http_headers()
        assert headers is not None
        assert headers["X-Glean-ActAs"] == "  user@example.com  "
