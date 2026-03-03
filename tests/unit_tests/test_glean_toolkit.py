import os
from unittest.mock import MagicMock, patch

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

        self.mock_people_glean_patcher = patch("langchain_glean.retrievers.people.Glean")
        self.mock_search_glean_patcher = patch("langchain_glean.retrievers.search.Glean")
        self.mock_people_glean_patcher.start()
        self.mock_search_glean_patcher.start()

        yield

        self.mock_people_glean_patcher.stop()
        self.mock_search_glean_patcher.stop()

        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_get_tools_returns_six_tools(self) -> None:
        """Test that get_tools returns exactly six tools."""
        tk = GleanToolkit()
        tools = tk.get_tools()

        assert len(tools) == 6

    def test_get_tools_contains_all_tool_types(self) -> None:
        """Test that get_tools returns one of each expected tool type."""
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

        for tool_type in expected_types:
            matching = [t for t in tools if isinstance(t, tool_type)]
            assert len(matching) == 1, f"Expected exactly one {tool_type.__name__}, found {len(matching)}"

    def test_get_tools_unique_names(self) -> None:
        """Test that all tools have unique names."""
        tk = GleanToolkit()
        tools = tk.get_tools()
        names = [t.name for t in tools]

        assert len(names) == len(set(names)), f"Duplicate tool names found: {names}"

    def test_search_tool_uses_provided_search_retriever(self) -> None:
        """Test that the GleanSearchTool uses the toolkit's search_retriever."""
        tk = GleanToolkit()
        tools = tk.get_tools()
        search_tool = next(t for t in tools if isinstance(t, GleanSearchTool))

        assert search_tool.retriever is tk.search_retriever

    def test_people_tool_uses_provided_people_retriever(self) -> None:
        """Test that the GleanPeopleProfileSearchTool uses the toolkit's people_retriever."""
        tk = GleanToolkit()
        tools = tk.get_tools()
        people_tool = next(t for t in tools if isinstance(t, GleanPeopleProfileSearchTool))

        assert people_tool.retriever is tk.people_retriever

    def test_custom_retrievers(self) -> None:
        """Test that custom retrievers can be passed to the toolkit."""
        from langchain_glean.retrievers.people import GleanPeopleProfileRetriever
        from langchain_glean.retrievers.search import GleanSearchRetriever

        custom_search = MagicMock(spec=GleanSearchRetriever)
        custom_people = MagicMock(spec=GleanPeopleProfileRetriever)

        tk = GleanToolkit(search_retriever=custom_search, people_retriever=custom_people)
        tools = tk.get_tools()

        search_tool = next(t for t in tools if isinstance(t, GleanSearchTool))
        people_tool = next(t for t in tools if isinstance(t, GleanPeopleProfileSearchTool))

        assert search_tool.retriever is custom_search
        assert people_tool.retriever is custom_people

    def test_toolkit_without_act_as(self) -> None:
        """Test that the toolkit works without GLEAN_ACT_AS set."""
        os.environ.pop("GLEAN_ACT_AS", None)

        tk = GleanToolkit()
        tools = tk.get_tools()

        assert len(tools) == 6
