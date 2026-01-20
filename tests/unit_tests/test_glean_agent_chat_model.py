import os
from typing import Any, Dict, List, Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_glean.chat_models.agent_chat import ChatGleanAgent


class TestGleanAgentChatModel:
    """Test the ChatGleanAgent model."""

    @property
    def model_class(self) -> Type[BaseChatModel]:
        """Return the model class to test."""
        return ChatGleanAgent

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """Return model kwargs to use for testing."""
        return {"agent_id": "test-agent-id"}

    @property
    def messages(self) -> List[BaseMessage]:
        """Return messages to use for testing."""
        return [
            HumanMessage(content="Hello, how are you?"),
        ]

    @property
    def messages_with_system(self) -> List[BaseMessage]:
        """Return messages with a system message to use for testing."""
        return [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Hello, how are you?"),
        ]

    @property
    def messages_with_chat_history(self) -> List[BaseMessage]:
        """Return messages with chat history to use for testing."""
        return [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="The capital of France is Paris."),
            HumanMessage(content="What is its population?"),
        ]

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test."""
        # Set environment variables for testing
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-api-token"

        # Mock the Glean class where it's directly used
        self.mock_glean_patcher = patch("langchain_glean.chat_models.agent_chat.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Mock the client property of the Glean instance
        mock_client = MagicMock()
        self.mock_glean.return_value.__enter__.return_value.client = mock_client

        # Create mock agents client
        mock_agents = MagicMock()

        # Create the mock response
        mock_message = {"author": "GLEAN_AI", "fragments": [{"text": "This is a mock response from Glean Agent."}]}

        # Configure the run method to return a mock response
        mock_response = MagicMock()
        mock_response.messages = [mock_message]

        # Set up both sync and async mocks with the same response
        mock_agents.run.return_value = mock_response

        # For async mocking, we'll create a coroutine that returns the mock_response
        async def mock_run_async(*args, **kwargs):
            return mock_response

        # Assign the mock coroutine to run_async
        mock_agents.run_async = mock_run_async

        mock_client.agents = mock_agents

        self.field_patcher = patch("langchain_glean.chat_models.agent_chat.Field", side_effect=lambda default=None, **kwargs: default)
        self.field_mock = self.field_patcher.start()

        self.chat_model = ChatGleanAgent(agent_id="test-agent-id")

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()
        self.field_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN"]:
            os.environ.pop(var, None)

    # ===== BASIC TESTS =====

    def test_initialization(self):
        """Test that the chat model initializes correctly."""
        assert self.chat_model is not None
        assert self.chat_model.agent_id == "test-agent-id"

    def test_initialization_with_missing_env_vars(self):
        """Test initialization with missing environment variables."""
        del os.environ["GLEAN_INSTANCE"]
        del os.environ["GLEAN_API_TOKEN"]

        with pytest.raises(ValueError):
            ChatGleanAgent(agent_id="test-agent-id")

    def test_extract_user_input(self):
        """Test the _extract_user_input method."""
        # Test with a single human message
        input_str = self.chat_model._extract_user_input([HumanMessage(content="Hello")])
        assert input_str == "Hello"

        # Test with multiple human messages
        input_str = self.chat_model._extract_user_input([HumanMessage(content="Hello"), HumanMessage(content="How are you?")])
        assert input_str == "Hello\nHow are you?"

        # Test with mixed message types
        input_str = self.chat_model._extract_user_input(
            [SystemMessage(content="System prompt"), HumanMessage(content="User message"), AIMessage(content="AI response")]
        )
        assert input_str == "User message"

    def test_generate(self):
        """Test generating a response from the chat model."""
        result = self.chat_model._generate(self.messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "This is a mock response from Glean Agent."

        # Verify the run method was called with correct parameters
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.assert_called_once_with(
            agent_id="test-agent-id", input={"input": "Hello, how are you?"}
        )

    def test_generate_with_custom_fields(self):
        """Test generating a response with custom fields."""
        custom_fields = {"input": "Custom input", "param1": "value1"}
        result = self.chat_model._generate(self.messages, fields=custom_fields)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "This is a mock response from Glean Agent."

        # Verify the run method was called with the custom fields
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.assert_called_once_with(agent_id="test-agent-id", input=custom_fields)

    def test_generate_with_error(self):
        """Test error handling in _generate."""
        from glean.api_client import errors

        # Mock GleanError with required raw_response
        mock_response = MagicMock()
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.side_effect = error

        with pytest.raises(ValueError) as exc_info:
            self.chat_model._generate(self.messages)

        assert "Glean client error" in str(exc_info.value)

    def test_generate_with_generic_exception(self):
        """Test generic exception handling in _generate."""
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.side_effect = Exception("Network error")

        result = self.chat_model._generate(self.messages)

        assert len(result.generations) == 1
        assert "(offline)" in result.generations[0].message.content
        assert "Unable to reach Glean" in result.generations[0].message.content

    def test_generate_with_stop_sequences(self):
        """Test that providing stop sequences raises an error."""
        with pytest.raises(ValueError) as exc_info:
            self.chat_model._generate(self.messages, stop=["STOP"])

        assert "stop sequences are not supported" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_agenerate(self):
        """Test async generating a response from the chat model."""

        # Override the run_async method to always return a successful response
        async def mock_run_async(*args, **kwargs):
            mock_message = {"author": "GLEAN_AI", "fragments": [{"text": "This is a mock response from Glean Agent."}]}
            mock_response = MagicMock()
            mock_response.messages = [mock_message]
            return mock_response

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async

        result = await self.chat_model._agenerate(self.messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "This is a mock response from Glean Agent."

    @pytest.mark.asyncio
    async def test_agenerate_with_custom_fields(self):
        """Test async generating a response with custom fields."""

        # Override the run_async method to always return a successful response
        async def mock_run_async(*args, **kwargs):
            mock_message = {"author": "GLEAN_AI", "fragments": [{"text": "This is a mock response from Glean Agent."}]}
            mock_response = MagicMock()
            mock_response.messages = [mock_message]
            return mock_response

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async

        custom_fields = {"input": "Custom input", "param1": "value1"}
        result = await self.chat_model._agenerate(self.messages, fields=custom_fields)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "This is a mock response from Glean Agent."

    @pytest.mark.asyncio
    async def test_agenerate_with_error(self):
        """Test error handling in _agenerate."""
        from glean.api_client import errors

        # Mock GleanError with required raw_response
        mock_response = MagicMock()
        error = errors.GleanError("Test error", raw_response=mock_response)

        # Override the run_async method to raise an error
        async def mock_run_async_error(*args, **kwargs):
            raise error

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async_error

        with pytest.raises(ValueError) as exc_info:
            await self.chat_model._agenerate(self.messages)

        assert "Glean client error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_agenerate_with_generic_exception(self):
        """Test generic exception handling in _agenerate."""

        # Override the run_async method to raise a generic exception
        async def mock_run_async_error(*args, **kwargs):
            raise Exception("Network error")

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async_error

        result = await self.chat_model._agenerate(self.messages)

        assert len(result.generations) == 1
        assert "(offline)" in result.generations[0].message.content
        assert "Unable to reach Glean" in result.generations[0].message.content

    @pytest.mark.asyncio
    async def test_agenerate_with_stop_sequences(self):
        """Test that providing stop sequences raises an error in _agenerate."""
        with pytest.raises(ValueError) as exc_info:
            await self.chat_model._agenerate(self.messages, stop=["STOP"])

        assert "stop sequences are not supported" in str(exc_info.value)
