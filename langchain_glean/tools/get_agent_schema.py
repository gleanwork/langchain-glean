from typing import Any

from glean.api_client import Glean, errors
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class _GetSchemaArgs(BaseModel):
    agent_id: str = Field(..., description="ID of the agent")


class GleanGetAgentSchemaTool(GleanAPIClientMixin, BaseTool):
    """Tool that retrieves the input schema for a specific agent."""

    name: str = "glean_get_agent_schema"
    description: str = "Fetch the input schema for a Glean agent using its ID."

    args_schema: type = _GetSchemaArgs

    def _run(self, agent_id: str, **kwargs: Any) -> str:  # noqa: D401
        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = g.client.agents.retrieve_schemas(agent_id=agent_id)

            if hasattr(response, "model_dump_json"):
                return response.model_dump_json(indent=2)

            return str(response)
        except errors.GleanError as e:
            details = f"Glean API error: {e}"
            if getattr(e, "raw_response", None):
                details += f": {e.raw_response}"
            return details
        except Exception as e:  # noqa: BLE001
            return f"Error getting agent schema: {e}"

    async def _arun(self, agent_id: str, **kwargs: Any) -> str:  # noqa: D401
        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = await g.client.agents.retrieve_schemas_async(agent_id=agent_id)

            if hasattr(response, "model_dump_json"):
                return response.model_dump_json(indent=2)

            return str(response)
        except errors.GleanError as e:
            details = f"Glean API error: {e}"
            if getattr(e, "raw_response", None):
                details += f": {e.raw_response}"
            return details
        except Exception as e:  # noqa: BLE001
            return f"Error getting agent schema: {e}"
