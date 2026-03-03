import os
from unittest.mock import patch

import pytest

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
        """Set up the test environment."""
        os.environ["GLEAN_INSTANCE"] = "test-glean"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "test@example.com"

        yield

        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_get_tools_count(self) -> None:
        """Test that the toolkit returns exactly 6 tools."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        assert len(tools) == 6

    def test_get_tools_contains_all_types(self) -> None:
        """Test that all expected tool types are present."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        expected_types = [
            GleanChatTool,
            GleanPeopleProfileSearchTool,
            GleanSearchTool,
            GleanListAgentsTool,
            GleanGetAgentSchemaTool,
            GleanRunAgentTool,
        ]
        for expected_type in expected_types:
            assert any(isinstance(t, expected_type) for t in tools), f"Missing tool type: {expected_type.__name__}"

    def test_get_tools_names(self) -> None:
        """Test that all tools have the expected names."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        tool_names = {t.name for t in tools}
        expected_names = {
            "chat",
            "people_profile_search",
            "glean_search",
            "glean_list_agents",
            "glean_get_agent_schema",
            "glean_run_agent",
        }
        assert tool_names == expected_names

    def test_get_tools_shares_retrievers(self) -> None:
        """Test that tools receive the toolkit's retriever instances."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        search_tool = next(t for t in tools if isinstance(t, GleanSearchTool))
        people_tool = next(t for t in tools if isinstance(t, GleanPeopleProfileSearchTool))

        assert search_tool.retriever is tk.search_retriever
        assert people_tool.retriever is tk.people_retriever

    def test_get_tools_returns_new_list_each_call(self) -> None:
        """Test that get_tools returns a new list on each call."""
        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools1 = tk.get_tools()
            tools2 = tk.get_tools()

        assert tools1 is not tools2
        assert len(tools1) == len(tools2)
