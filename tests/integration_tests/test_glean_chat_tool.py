import os
import unittest
from typing import Type

from dotenv import load_dotenv

from langchain_glean.chat_models.chat import ChatBasicRequest
from langchain_glean.tools.chat import GleanChatTool


class TestGleanChatTool(unittest.TestCase):
    """Test the GleanChatTool with actual API calls."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def tool_constructor(self) -> Type[GleanChatTool]:
        """Get the tool constructor for integration tests."""
        return GleanChatTool

    @property
    def tool_constructor_params(self) -> dict:
        """Get the parameters for the tool constructor."""
        return {}

    @property
    def basic_message_example(self) -> str:
        """Returns a simple message example for the tool."""
        return "What can Glean's assistant do?"

    @property
    def basic_request_example(self) -> ChatBasicRequest:
        """Returns a ChatBasicRequest example."""
        return ChatBasicRequest(
            message="Summarize what Glean is", context=["Glean is an enterprise search platform.", "It uses AI to provide relevant search results."]
        )

    def test_invoke_with_simple_message(self) -> None:
        """Test invoking with a simple message string."""
        tool = self.tool_constructor(**self.tool_constructor_params)
        output = tool.invoke(input=self.basic_message_example)

        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    def test_invoke_with_basic_request(self) -> None:
        """Test invoking with a ChatBasicRequest object."""
        tool = self.tool_constructor(**self.tool_constructor_params)
        chat_request = self.basic_request_example

        output = tool.invoke(input=chat_request.message)

        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    def test_invoke_with_advanced_params(self) -> None:
        """Test invoking with advanced parameters."""
        tool = self.tool_constructor(**self.tool_constructor_params)

        output = tool.invoke(input=self.basic_message_example)

        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    def test_async_invoke_with_simple_message(self) -> None:
        """Test async invoking with a simple message string."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)
            output = await tool.ainvoke(input=self.basic_message_example)

            self.assertIsInstance(output, str)
            self.assertTrue(len(output) > 0)

        asyncio.run(_test())

    def test_async_invoke_with_basic_request(self) -> None:
        """Test async invoking with a ChatBasicRequest object."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)
            chat_request = self.basic_request_example

            output = await tool.ainvoke(input=chat_request.message)

            self.assertIsInstance(output, str)
            self.assertTrue(len(output) > 0)

        asyncio.run(_test())

    def test_async_invoke_with_advanced_params(self) -> None:
        """Test async invoking with advanced parameters."""
        import asyncio

        async def _test():
            tool = self.tool_constructor(**self.tool_constructor_params)

            output = await tool.ainvoke(input=self.basic_message_example)

            self.assertIsInstance(output, str)
            self.assertTrue(len(output) > 0)

        asyncio.run(_test())
