import os
import unittest
from typing import Type

from dotenv import load_dotenv

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool


class TestGleanSearchTool(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_SUBDOMAIN") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def tool_constructor(self) -> Type[GleanSearchTool]:
        """Get the tool constructor for integration tests."""
        return GleanSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        """Get the parameters for the tool constructor."""

        retriever = GleanSearchRetriever()
        return {"retriever": retriever}

    @property
    def tool_input_example(self) -> dict:
        """Returns an example input for the tool."""
        return {
            "query": "example query",
            "page_size": 100,
            "disable_spellcheck": True,
            "request_options": {"facetFilters": [{"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}]}]},
        }

    def test_invoke_matches_output_schema(self) -> None:
        """Test that invoke returns output matching the output schema."""
        tool = self.tool_constructor(**self.tool_constructor_params)
        output = tool.invoke(self.tool_input_example["query"])
        self.assertIsInstance(output, str)

    def test_invoke_no_tool_call(self) -> None:
        """Test that invoke works without a tool call."""
        tool = self.tool_constructor(**self.tool_constructor_params)
        output = tool.invoke(self.tool_input_example["query"])
        self.assertIsInstance(output, str)

    def test_async_invoke_matches_output_schema(self) -> None:
        """Test that async_invoke returns output matching the output schema."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)
            output = await tool.ainvoke(self.tool_input_example["query"])
            self.assertIsInstance(output, str)

        asyncio.run(_test())

    def test_async_invoke_no_tool_call(self) -> None:
        """Test that async_invoke works without a tool call."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)
            output = await tool.ainvoke(self.tool_input_example["query"])
            self.assertIsInstance(output, str)

        asyncio.run(_test())
