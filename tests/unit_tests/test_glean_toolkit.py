import os
from unittest.mock import patch

from langchain_glean.toolkit import GleanToolkit
from langchain_glean.tools.chat import GleanChatTool
from langchain_glean.tools.people_profile_search import GleanPeopleProfileSearchTool
from langchain_glean.tools.search import GleanSearchTool


class TestGleanToolkit:
    """Verify that the toolkit returns the expected tools."""

    def test_get_tools(self) -> None:
        os.environ["GLEAN_INSTANCE"] = "test-glean"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "test@example.com"

        with patch("langchain_glean.retrievers.people.Glean"), patch("langchain_glean.retrievers.search.Glean"):
            tk = GleanToolkit()
            tools = tk.get_tools()

        assert len(tools) == 6
        assert any(isinstance(t, GleanChatTool) for t in tools)
        assert any(isinstance(t, GleanPeopleProfileSearchTool) for t in tools)
        assert any(isinstance(t, GleanSearchTool) for t in tools)

        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)
