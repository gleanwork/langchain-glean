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

from langchain_glean.chat_models.chat import ChatBasicRequest, ChatGlean


class TestGleanChatModel:
    """Test the ChatGlean model."""

    @property
    def model_class(self) -> Type[BaseChatModel]:
        """Return the model class to test."""
        return ChatGlean

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """Return model kwargs to use for testing."""
        return {}

    @property
    def messages(self) -> List[BaseMessage]:
        """Return messages to use for testing."""
        return [
            SystemMessage(content="You are a helpful AI assistant."),
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
        self.mock_glean_patcher = patch("langchain_glean.chat_models.chat.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Mock the client property of the Glean instance
        mock_client = MagicMock()
        self.mock_glean.return_value.__enter__.return_value.client = mock_client

        # Create mock chat client
        mock_chat = MagicMock()
        from types import SimpleNamespace

        # Configure chat.create to mimic expected Glean response structure
        mock_response = SimpleNamespace(
            messages=[{"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "mock"}]}],
            chatId="mock-chat-id",
            chatSessionTrackingToken="token",
        )
        mock_chat.create = MagicMock(return_value=mock_response)
        mock_client.chat = mock_chat

        # Create mock ChatMessage object for the response
        mock_fragment = MagicMock()
        mock_fragment.text = "This is a mock response from Glean AI."

        mock_message = MagicMock()
        mock_message.author = "GLEAN_AI"
        mock_message.message_type = "CONTENT"
        mock_message.fragments = [mock_fragment]

        # Convert this to a Dict for the AI message extraction in _generate
        mock_message_dict = {"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "This is a mock response from Glean AI."}]}

        # Mock the create method
        mock_response = MagicMock()
        mock_response.messages = [mock_message_dict]  # Use dict format for the response
        mock_response.chatId = "mock-chat-id"
        mock_response.chatSessionTrackingToken = "mock-tracking-token"
        mock_chat.create.return_value = mock_response
        mock_chat.create_async.return_value = mock_response

        # Mock the create_stream method for streaming responses
        mock_stream = "{"
        mock_stream += '"chatId": "mock-chat-id", "chatSessionTrackingToken": "mock-tracking-token", "messages": []'
        mock_stream += "}\n{"
        mock_stream += '"messages": [{"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "This is "}]}]'
        mock_stream += "}\n{"
        mock_stream += '"messages": [{"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "a streaming response."}]}]'
        mock_stream += "}"
        mock_chat.create_stream.return_value = mock_stream
        mock_chat.create_stream_async.return_value = mock_stream

        self.field_patcher = patch("langchain_glean.chat_models.chat.Field", side_effect=lambda default=None, **kwargs: default)
        self.field_mock = self.field_patcher.start()

        self.chat_model = ChatGlean()

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()
        self.field_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    # ===== BASIC TESTS =====

    def test_initialization(self):
        """Test that the chat model initializes correctly."""
        assert self.chat_model is not None

    def test_initialization_with_missing_env_vars(self):
        """Test initialization with missing environment variables."""
        del os.environ["GLEAN_INSTANCE"]
        del os.environ["GLEAN_API_TOKEN"]

        with pytest.raises(ValueError):
            ChatGlean()

    def test_convert_message_to_glean_format(self):
        """Test converting LangChain messages to Glean format."""
        human_msg = HumanMessage(content="Hello, Glean!")
        glean_msg = self.chat_model._convert_message_to_glean_format(human_msg)

        # Check attributes instead of dictionary access
        assert glean_msg.author == "USER"
        assert glean_msg.message_type == "CONTENT"
        assert glean_msg.fragments[0].text == "Hello, Glean!"

        ai_msg = AIMessage(content="Hello, human!")
        glean_msg = self.chat_model._convert_message_to_glean_format(ai_msg)
        assert glean_msg.author == "GLEAN_AI"
        assert glean_msg.message_type == "CONTENT"
        assert glean_msg.fragments[0].text == "Hello, human!"

        system_msg = SystemMessage(content="You are an AI assistant.")
        glean_msg = self.chat_model._convert_message_to_glean_format(system_msg)
        assert glean_msg.author == "USER"
        assert glean_msg.message_type == "CONTEXT"
        assert glean_msg.fragments[0].text == "You are an AI assistant."

    def test_generate(self):
        """Test generating a response from the chat model."""
        # Create a mock ChatRequest
        mock_request = MagicMock()

        with (
            patch("langchain_glean.chat_models.chat.models.ChatRequest", return_value=mock_request),
            patch("langchain_glean.chat_models.chat.models.ChatMessage"),
            patch("langchain_glean.chat_models.chat.models.AgentConfig"),
            patch.object(self.chat_model, "_convert_glean_message_to_langchain") as mock_convert,
        ):
            # Mock the convert method to return a message with specific content
            mock_convert.return_value = AIMessage(content="This is a mock response from Glean AI.")

            result = self.chat_model._generate(self.messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "This is a mock response from Glean AI."
        assert result.generations[0].generation_info["chat_id"] == "mock-chat-id"
        assert result.generations[0].generation_info["tracking_token"] == "mock-tracking-token"

        assert self.chat_model.chat_id == "mock-chat-id"
        self.mock_glean.return_value.__enter__.return_value.client.chat.create.assert_called_once()

    # ===== ADVANCED TESTS =====

    def test_invoke_with_basic_request(self):
        """Test invoking with a ChatBasicRequest object."""
        with (
            patch.object(self.chat_model, "_generate") as mock_generate,
            patch.object(self.chat_model, "_messages_from_chat_input") as mock_messages_from_chat_input,
        ):
            # Mock the _generate method to return a simple result
            mock_result = MagicMock()
            mock_result.generations = [MagicMock()]
            mock_result.generations[0].message = AIMessage(content="Test response")
            mock_generate.return_value = mock_result

            # Mock _messages_from_chat_input to return two messages
            # (this is how the actual method is implemented - converts the request into a system message and a human message)
            expected_messages = [
                SystemMessage(content="Glean is an enterprise search platform.\nIt uses AI to provide better search results."),
                HumanMessage(content="What is Glean?"),
            ]
            mock_messages_from_chat_input.return_value = expected_messages

            # Create a ChatBasicRequest
            request = ChatBasicRequest(
                message="What is Glean?", context=["Glean is an enterprise search platform.", "It uses AI to provide better search results."]
            )

            result = self.chat_model.invoke(request)

            # Check that _messages_from_chat_input was called with the request
            mock_messages_from_chat_input.assert_called_once_with(request)

            # Check that _generate was called correctly
            mock_generate.assert_called_once()
            args = mock_generate.call_args[0]
            assert args[0] == expected_messages

    async def test_ainvoke_with_basic_request(self):
        """Test async invoking with a ChatBasicRequest object."""
        with (
            patch.object(self.chat_model, "_agenerate") as mock_agenerate,
            patch.object(self.chat_model, "_messages_from_chat_input") as mock_messages_from_chat_input,
        ):
            # Mock the _agenerate method to return a simple result
            mock_result = MagicMock()
            mock_result.generations = [MagicMock()]
            mock_result.generations[0].message = AIMessage(content="Test async response")
            mock_agenerate.return_value = mock_result

            # Mock _messages_from_chat_input to return expected messages
            expected_messages = [
                SystemMessage(content="Glean is an enterprise search platform.\nIt uses AI to provide better search results."),
                HumanMessage(content="What is Glean?"),
            ]
            mock_messages_from_chat_input.return_value = expected_messages

            # Create a ChatBasicRequest
            request = ChatBasicRequest(
                message="What is Glean?", context=["Glean is an enterprise search platform.", "It uses AI to provide better search results."]
            )

            result = await self.chat_model.ainvoke(request)

            # Check that _messages_from_chat_input was called with the request
            mock_messages_from_chat_input.assert_called_once_with(request)

            # Check that _agenerate was called correctly
            mock_agenerate.assert_called_once()
            args = mock_agenerate.call_args[0]
            assert args[0] == expected_messages

            # Verify the result content
            assert result.content == "Test async response"

    def test_invoke_with_advanced_params(self):
        """Test invoking with advanced Glean parameters."""
        with patch.object(self.chat_model, "_generate") as mock_generate:
            # Mock the _generate method to return a simple result
            mock_result = MagicMock()
            mock_result.generations = [MagicMock()]
            mock_result.generations[0].message = AIMessage(content="Test response")
            mock_generate.return_value = mock_result

            # Advanced parameters to pass through
            params = {
                "save_chat": True,
                "agent_config": {"agent": "COPILOT", "mode": "ANSWER"},
                "timeout_millis": 30000,
                "inclusions": {"url_patterns": ["https://glean.com/*"]},
                "exclusions": {"url_patterns": ["https://glean.com/blog/*"]},
            }

            # Call with advanced parameters
            result = self.chat_model.invoke(self.messages, **params)

            # Verify _generate was called with the right params
            mock_generate.assert_called_once()
            kwargs = mock_generate.call_args[1]
            assert kwargs["save_chat"] is True
            assert kwargs["agent_config"]["agent"] == "COPILOT"
            assert kwargs["agent_config"]["mode"] == "ANSWER"
            assert kwargs["timeout_millis"] == 30000
            assert kwargs["inclusions"]["url_patterns"] == ["https://glean.com/*"]
            assert kwargs["exclusions"]["url_patterns"] == ["https://glean.com/blog/*"]
