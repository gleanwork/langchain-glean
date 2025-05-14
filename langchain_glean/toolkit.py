from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field

from langchain_glean.retrievers.people import GleanPeopleProfileRetriever
from langchain_glean.retrievers.search import GleanSearchRetriever
from langchain_glean.tools.chat import GleanChatTool
from langchain_glean.tools.get_agent_schema import GleanGetAgentSchemaTool
from langchain_glean.tools.list_agents import GleanListAgentsTool
from langchain_glean.tools.people_profile_search import (
    GleanPeopleProfileSearchTool,
)
from langchain_glean.tools.run_agent import GleanRunAgentTool
from langchain_glean.tools.search import GleanSearchTool


class GleanToolkit(BaseToolkit):
    """Aggregates all first-party Glean tools for easy agent injection."""

    search_retriever: GleanSearchRetriever = Field(default_factory=GleanSearchRetriever, repr=False)
    people_retriever: GleanPeopleProfileRetriever = Field(default_factory=GleanPeopleProfileRetriever, repr=False)

    def get_tools(self) -> List[BaseTool]:
        """Return instantiated Glean tools."""
        return [
            GleanChatTool(),
            GleanPeopleProfileSearchTool(retriever=self.people_retriever),
            GleanSearchTool(retriever=self.search_retriever),
            GleanListAgentsTool(),
            GleanGetAgentSchemaTool(),
            GleanRunAgentTool(),
        ]
