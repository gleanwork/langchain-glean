from typing import Any, Dict, Optional

from glean import Glean
from glean.client import Client
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, PrivateAttr, model_post_init, model_validator


class GleanAPIClientMixin:  # noqa: D401
    """Shared auth + client bootstrap for Glean wrappers.

    Adds three public config fields (``instance``, ``api_token``, ``act_as``) and
    instantiates :pyclass:`glean.Glean` on model construction.  The resulting
    client is available via :pyattr:`client`.
    """

    instance: str = Field(description="Glean instance/subdomain (e.g. 'acme')")
    api_token: str = Field(description="Glean API token (user or global)")
    act_as: Optional[str] = Field(
        default=None,
        description="Email to act as when using a global token. Ignored for user tokens.",
    )

    client: Client = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401, ANN001
        values = values or {}
        values["instance"] = get_from_dict_or_env(values, "instance", "GLEAN_INSTANCE")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")
        return values

    @model_post_init
    def _init_client(self, __context: Any) -> None:  # noqa: D401
        try:
            g = Glean(api_token=self.api_token, instance=self.instance)
            self.client = g.client
        except Exception as exc:
            raise ValueError(f"Failed to initialize Glean client: {exc}") from exc 