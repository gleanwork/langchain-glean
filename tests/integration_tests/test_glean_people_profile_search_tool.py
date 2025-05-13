import os
import unittest
from typing import Type

from dotenv import load_dotenv

from langchain_glean.retrievers.people import GleanPeopleProfileRetriever, PeopleProfileBasicRequest
from langchain_glean.tools.people_profile_search import GleanPeopleProfileSearchTool


class TestGleanPeopleProfileSearchTool(unittest.TestCase):
    """Test the GleanPeopleProfileSearchTool with actual API calls."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def tool_constructor(self) -> Type[GleanPeopleProfileSearchTool]:
        """Get the tool constructor for integration tests."""
        return GleanPeopleProfileSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        """Get the parameters for the tool constructor."""
        retriever = GleanPeopleProfileRetriever()
        return {"retriever": retriever}

    @property
    def string_query_example(self) -> str:
        """Returns an example query string for the tool."""
        return "engineer"

    @property
    def typed_request_example(self) -> PeopleProfileBasicRequest:
        """Returns a typed request example for the tool."""
        return PeopleProfileBasicRequest(query="manager", page_size=5)

    @property
    def filtered_request_example(self) -> PeopleProfileBasicRequest:
        """Returns a request with filters example."""

        return PeopleProfileBasicRequest(filters={"department": "Engineering"}, page_size=5)

    def test_invoke_with_string_query(self) -> None:
        """Test invoking with a string query."""
        tool = self.tool_constructor(**self.tool_constructor_params)

        output = tool.invoke(input=self.string_query_example)

        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0 and (output != "No matching people found." or "- " in output))

    def test_invoke_with_typed_request(self) -> None:
        """Test invoking with a typed request."""
        tool = self.tool_constructor(**self.tool_constructor_params)
        request = self.typed_request_example

        output = tool.invoke(input=request.query)

        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0 and (output != "No matching people found." or "- " in output))

    def test_invoke_with_filters(self) -> None:
        """Test invoking with filters."""
        tool = self.tool_constructor(**self.tool_constructor_params)
        output = tool.invoke(input="software engineer in engineering")

        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0 and (output != "No matching people found." or "- " in output))

    def test_async_invoke_with_string_query(self) -> None:
        """Test async invoking with a string query."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)
            output = await tool.ainvoke(input=self.string_query_example)

            self.assertIsInstance(output, str)
            self.assertTrue(len(output) > 0 and (output != "No matching people found." or "- " in output))

        asyncio.run(_test())

    def test_async_invoke_with_typed_request(self) -> None:
        """Test async invoking with a typed request."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)
            request = self.typed_request_example

            output = await tool.ainvoke(input=request.query)

            self.assertIsInstance(output, str)
            self.assertTrue(len(output) > 0 and (output != "No matching people found." or "- " in output))

        asyncio.run(_test())

    def test_async_invoke_with_filters(self) -> None:
        """Test async invoking with filters."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)
            output = await tool.ainvoke(input="software engineer in engineering")

            self.assertIsInstance(output, str)
            self.assertTrue(len(output) > 0 and (output != "No matching people found." or "- " in output))

        asyncio.run(_test())
