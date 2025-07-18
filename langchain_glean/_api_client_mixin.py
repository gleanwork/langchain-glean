from typing import Any, Dict, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, model_validator


class GleanAPIClientMixin:  # noqa: D401
    """Shared auth + client bootstrap for Glean wrappers.

    Provides configuration for creating Glean API clients.
    """

    instance: str = Field(description="Glean instance/subdomain (e.g. 'acme')")
    api_token: str = Field(description="Glean API token (user or global)")
    act_as: Optional[str] = Field(
        default=None,
        description="Email to act as when using a global token. Ignored for user tokens.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401, ANN001
        values = values or {}
        values["instance"] = get_from_dict_or_env(values, "instance", "GLEAN_INSTANCE")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")
        return values

    def _http_headers(self) -> Optional[Dict[str, str]]:
        """Return HTTP headers for impersonation if ``act_as`` is set."""
        return {"X-Glean-ActAs": str(self.act_as)} if self.act_as else None
