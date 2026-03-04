import os
from typing import Any, Dict, Optional

from glean.api_client import Glean
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, model_validator


class GleanAPIClientMixin:  # noqa: D401
    """Shared auth + client bootstrap for Glean wrappers.

    Provides configuration for creating Glean API clients.
    """

    server_url: Optional[str] = Field(
        default=None,
        description="Full Glean backend URL (e.g. 'https://acme-be.glean.com'). Preferred over instance. Falls back to GLEAN_SERVER_URL env var.",
    )
    instance: str = Field(
        default="",
        description="Glean instance/subdomain (e.g. 'acme'). Legacy — prefer server_url. Falls back to GLEAN_INSTANCE env var.",
    )
    api_token: str = Field(description="Glean API token (user or global)")
    act_as: Optional[str] = Field(
        default=None,
        description="Email to act as when using a global token. Ignored for user tokens.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401, ANN001
        values = values or {}
        if not values.get("server_url"):
            values["server_url"] = os.environ.get("GLEAN_SERVER_URL", "")
        if not values.get("server_url") and not values.get("instance"):
            values["instance"] = get_from_dict_or_env(values, "instance", "GLEAN_INSTANCE")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")
        return values

    def _build_glean_client(self) -> Glean:
        """Create a Glean SDK client using server_url (preferred) or instance."""
        if self.server_url:
            return Glean(server_url=self.server_url, api_token=self.api_token)
        return Glean(instance=self.instance, api_token=self.api_token)

    def _http_headers(self) -> Optional[Dict[str, str]]:
        """Return HTTP headers for impersonation if ``act_as`` is set."""
        return {"X-Glean-ActAs": str(self.act_as)} if self.act_as else None
