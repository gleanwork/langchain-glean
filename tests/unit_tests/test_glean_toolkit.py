import os
from unittest.mock import patch

import pytest
from langchain_core.tools import BaseTool

from langchain_glean.toolkit import GleanToolkit
from langchain_glean.tools.chat import GleanChatTool
from langchain_glean.tools.get_agent_schema import GleanGetAgentSchemaTool
from langchain_glean.tools.list_agents import GleanListAgentsTool
from langchain_glean.tools.people_profile_search import GleanPeopleProfileSearchTool
from langchain_glean.tools.run_agent import GleanRunAgentTool
from langchain_glean.tools.search import GleanSearchTool


class TestGleanToolkit:
    """Verify that the toolkit returns the expected tools."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up env vars and mocks for toolkit tests."""
        os.environ["GLEAN_INSTANCE"] = "test-glean"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "test@example.com"

        yield

        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_get_tools(self) -> None:
        """Test that the toolkit returns all 6 tools."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        assert len(tools) == 6
        assert any(isinstance(t, GleanChatTool) for t in tools)
        assert any(isinstance(t, GleanPeopleProfileSearchTool) for t in tools)
        assert any(isinstance(t, GleanSearchTool) for t in tools)

    def test_get_tools_contains_all_tool_types(self) -> None:
        """Verify every expected tool type is present."""
        expected_types = {
            GleanChatTool,
            GleanSearchTool,
            GleanPeopleProfileSearchTool,
            GleanListAgentsTool,
            GleanGetAgentSchemaTool,
            GleanRunAgentTool,
        }

        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        actual_types = {type(t) for t in tools}
        assert actual_types == expected_types

    def test_tools_are_base_tool_instances(self) -> None:
        """All returned tools should be BaseTool instances."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        for tool in tools:
            assert isinstance(tool, BaseTool)

    def test_tools_have_unique_names(self) -> None:
        """Every tool should have a unique name."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        names = [t.name for t in tools]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_toolkit_shares_retrievers_with_tools(self) -> None:
        """Search and people tools should use the toolkit's retriever instances."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        search_tool = next(t for t in tools if isinstance(t, GleanSearchTool))
        people_tool = next(t for t in tools if isinstance(t, GleanPeopleProfileSearchTool))

        assert search_tool.retriever is tk.search_retriever
        assert people_tool.retriever is tk.people_retriever
