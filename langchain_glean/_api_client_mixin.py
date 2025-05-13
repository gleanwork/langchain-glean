from typing import Any, Dict, Optional

from glean import Glean
from glean.client import Client
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, PrivateAttr, model_validator


class GleanAPIClientMixin:  # noqa: D401
    """Shared auth + client bootstrap for Glean wrappers.

    Exposes a fully-authenticated :pyattr:`client` ready for querying.
    """

    instance: str = Field(description="Glean instance/subdomain (e.g. 'acme')")
    api_token: str = Field(description="Glean API token (user or global)")
    act_as: Optional[str] = Field(
        default=None,
        description="Email to act as when using a global token. Ignored for user tokens.",
    )

    client: Client = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401, ANN001
        values = values or {}
        values["instance"] = get_from_dict_or_env(values, "instance", "GLEAN_INSTANCE")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")
        return values

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]  # noqa: D401
        try:
            g = Glean(api_token=self.api_token, instance=self.instance)  # type: ignore[call-arg]
            self.client = g.client  # type: ignore[assignment]
        except Exception as exc:
            raise ValueError(f"Failed to initialize Glean client: {exc}") from exc
