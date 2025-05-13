import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from langchain_glean.chat_models.chat import ChatBasicRequest, ChatGlean
from langchain_glean.tools.chat import GleanChatTool


class TestGleanChatTool:
    """Test the GleanChatTool class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        # Set environment variables for testing
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-api-token"

        # Mock the ChatGlean class
        self.mock_chat_glean_patcher = patch("langchain_glean.tools.chat.ChatGlean")
        self.mock_chat_glean = self.mock_chat_glean_patcher.start()

        # Create mock instance of ChatGlean
        self.mock_chat_instance = MagicMock(spec=ChatGlean)
        self.mock_chat_glean.return_value = self.mock_chat_instance

        # Set up mock response for both invoke and ainvoke
        mock_response = AIMessage(content="This is a test response from Glean Assistant.")
        self.mock_chat_instance.invoke.return_value = mock_response
        self.mock_chat_instance.ainvoke.return_value = mock_response

        # Initialize the tool
        self.tool = GleanChatTool()

        yield

        # Clean up after tests
        self.mock_chat_glean_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        assert self.tool.name == "chat"
        assert self.tool.description == "Interact with Glean's AI assistant using a message and optional context."
        assert self.tool.args_schema == ChatBasicRequest

    def test_run_with_simple_message(self) -> None:
        """Test _run with a simple message string."""
        message = "What is Glean?"
        result = self.tool._run(message)

        # Verify that ChatGlean was initialized
        self.mock_chat_glean.assert_called_once()

        # Verify that invoke was called with the correct parameters
        self.mock_chat_instance.invoke.assert_called_once()
        call_args = self.mock_chat_instance.invoke.call_args
        first_arg = call_args[0][0]

        # First argument should be the message string
        assert first_arg == message

        assert result == "This is a test response from Glean Assistant."

    def test_run_with_message_and_context(self) -> None:
        """Test _run with a message and context."""
        message = "Summarize this information"
        context = ["Glean is a search platform.", "It uses AI to provide better results."]

        result = self.tool._run({"message": message, "context": context})

        # Verify that invoke was called with the correct parameters
        self.mock_chat_instance.invoke.assert_called_once()
        call_args = self.mock_chat_instance.invoke.call_args
        chat_request = call_args[0][0]

        assert isinstance(chat_request, ChatBasicRequest)
        assert chat_request.message == message
        assert chat_request.context == context

        assert result == "This is a test response from Glean Assistant."

    def test_run_with_additional_kwargs(self) -> None:
        """Test _run with additional keyword arguments."""
        message = "What is Glean?"

        # Additional kwargs that would be passed to the ChatGlean.invoke method
        result = self.tool._run({"message": message, "save_chat": True, "agent": "GPT", "mode": "QUICK"})

        # Verify that invoke was called with the correct parameters
        self.mock_chat_instance.invoke.assert_called_once()
        call_args = self.mock_chat_instance.invoke.call_args

        # Check the kwargs were passed through
        assert call_args[1]["save_chat"] is True
        assert call_args[1]["agent"] == "GPT"
        assert call_args[1]["mode"] == "QUICK"

        assert result == "This is a test response from Glean Assistant."

    async def test_arun_with_simple_message(self) -> None:
        """Test _arun with a simple message string."""
        message = "What is Glean?"
        result = await self.tool._arun(message)

        # Verify that ChatGlean was initialized
        self.mock_chat_glean.assert_called_once()

        # Verify that ainvoke was called with the correct parameters
        self.mock_chat_instance.ainvoke.assert_called_once()
        call_args = self.mock_chat_instance.ainvoke.call_args
        first_arg = call_args[0][0]

        # First argument should be the message string
        assert first_arg == message

        assert result == "This is a test response from Glean Assistant."

    async def test_arun_with_message_and_context(self) -> None:
        """Test _arun with a message and context."""
        message = "Summarize this information"
        context = ["Glean is a search platform.", "It uses AI to provide better results."]

        result = await self.tool._arun({"message": message, "context": context})

        # Verify that ainvoke was called with the correct parameters
        self.mock_chat_instance.ainvoke.assert_called_once()
        call_args = self.mock_chat_instance.ainvoke.call_args
        chat_request = call_args[0][0]

        assert isinstance(chat_request, ChatBasicRequest)
        assert chat_request.message == message
        assert chat_request.context == context

        assert result == "This is a test response from Glean Assistant."

    async def test_arun_with_additional_kwargs(self) -> None:
        """Test _arun with additional keyword arguments."""
        message = "What is Glean?"

        # Additional kwargs that would be passed to the ChatGlean.ainvoke method
        result = await self.tool._arun({"message": message, "save_chat": True, "agent": "GPT", "mode": "QUICK"})

        # Verify that ainvoke was called with the correct parameters
        self.mock_chat_instance.ainvoke.assert_called_once()
        call_args = self.mock_chat_instance.ainvoke.call_args

        # Check the kwargs were passed through
        assert call_args[1]["save_chat"] is True
        assert call_args[1]["agent"] == "GPT"
        assert call_args[1]["mode"] == "QUICK"

        assert result == "This is a test response from Glean Assistant."

    def test_run_with_error(self) -> None:
        """Test _run when an error occurs."""
        # Make the mock raise an exception
        self.mock_chat_instance.invoke.side_effect = Exception("Test error")

        # The tool should catch the exception and return an error message
        message = "What is Glean?"
        result = self.tool._run({"message": message})

        assert "Error running Glean chat" in result
        assert "Test error" in result

    async def test_arun_with_error(self) -> None:
        """Test _arun when an error occurs."""
        # Make the mock raise an exception
        self.mock_chat_instance.ainvoke.side_effect = Exception("Test error")

        # The tool should catch the exception and return an error message
        message = "What is Glean?"
        result = await self.tool._arun({"message": message})

        assert "Error running Glean chat" in result
        assert "Test error" in result
