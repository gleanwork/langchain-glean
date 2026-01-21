from typing import Any, Dict

from glean.api_client import Glean, errors
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class RunAgentArgs(BaseModel):
    agent_id: str = Field(..., description="ID of the agent to run")
    fields: Dict[str, str] = Field(default_factory=dict, description="Input fields mapping for the agent")


class GleanRunAgentTool(GleanAPIClientMixin, BaseTool):
    """Tool that runs a specific agent with provided fields."""

    name: str = "glean_run_agent"
    description: str = "Run a Glean agent by ID with specified input fields."
    return_direct: bool = True

    args_schema: type = RunAgentArgs

    def _run(self, agent_id: str, fields: Dict[str, str], **kwargs: Any) -> str:  # noqa: D401
        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = g.client.agents.run(agent_id=agent_id, input=fields)

            if hasattr(response, "model_dump_json"):
                return response.model_dump_json(indent=2)

            return str(response)
        except errors.GleanError as e:
            details = f"Glean API error: {e}"
            if getattr(e, "raw_response", None):
                details += f": {e.raw_response}"
            return details
        except Exception as e:  # noqa: BLE001
            return f"Error running agent: {e}"

    async def _arun(self, agent_id: str, fields: Dict[str, str], **kwargs: Any) -> str:  # noqa: D401
        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = await g.client.agents.run_async(agent_id=agent_id, input=fields)

            if hasattr(response, "model_dump_json"):
                return response.model_dump_json(indent=2)

            return str(response)
        except errors.GleanError as e:
            details = f"Glean API error: {e}"
            if getattr(e, "raw_response", None):
                details += f": {e.raw_response}"
            return details
        except Exception as e:  # noqa: BLE001
            return f"Error running agent: {e}"
