from typing import Any

from glean.api_client import Glean, errors
from langchain_core.tools import BaseTool

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class GleanListAgentsTool(GleanAPIClientMixin, BaseTool):
    """Tool that lists available agents in a Glean instance."""

    name: str = "glean_list_agents"
    description: str = "List available Glean agents including their metadata."

    def _run(self, **kwargs: Any) -> str:  # noqa: D401
        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = g.client.agents.list()

            if hasattr(response, "model_dump_json"):
                return response.model_dump_json(indent=2)

            return str(response)
        except errors.GleanError as e:
            details = f"Glean API error: {e}"
            if getattr(e, "raw_response", None):
                details += f": {e.raw_response}"
            return details
        except Exception as e:  # noqa: BLE001
            return f"Error listing agents: {e}"

    async def _arun(self, **kwargs: Any) -> str:  # noqa: D401
        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = await g.client.agents.list_async()

            if hasattr(response, "model_dump_json"):
                return response.model_dump_json(indent=2)

            return str(response)
        except errors.GleanError as e:
            details = f"Glean API error: {e}"
            if getattr(e, "raw_response", None):
                details += f": {e.raw_response}"
            return details
        except Exception as e:  # noqa: BLE001
            return f"Error listing agents: {e}"
