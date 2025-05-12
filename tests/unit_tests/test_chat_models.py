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

from langchain_glean.chat_models import ChatGlean


class TestChatGlean:
    """Test the ChatGlean."""

    @property
    def model_class(self) -> Type[BaseChatModel]:
        """Return the model class to test."""
        return ChatGlean

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """Return model kwargs to use for testing."""
        return {}

    @property
    def model_unit_kwargs(self) -> Dict[str, Any]:
        """Return model kwargs to use for unit testing."""
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

        # Mock the Glean class
        self.mock_glean_patcher = patch("langchain_glean.chat_models.chat.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Mock the client property of the Glean instance
        mock_client = MagicMock()
        self.mock_glean.return_value.client = mock_client

        # Create mock chat client
        mock_chat = MagicMock()
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
        self.chat_model._client = mock_client

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()
        self.field_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_initialization(self):
        """Test that the chat model initializes correctly."""
        assert self.chat_model is not None
        assert hasattr(self.chat_model, "_client")

        self.mock_glean.assert_called_once_with(
            api_token="test-api-token",
            instance="test-instance",
        )

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

    def test_create_chat_request(self):
        """Test creating a chat request from messages."""
        messages = [SystemMessage(content="You are a helpful AI assistant."), HumanMessage(content="Hello, Glean!")]

        # Mock the models.ChatMessage and models.AgentConfig classes
        mock_chat_message = MagicMock()
        mock_agent_config = MagicMock()

        with (
            patch("langchain_glean.chat_models.chat.models.ChatMessage", return_value=mock_chat_message),
            patch("langchain_glean.chat_models.chat.models.AgentConfig", return_value=mock_agent_config),
            patch("langchain_glean.chat_models.chat.models.ChatRequest") as mock_chat_request,
        ):
            self.chat_model.save_chat = False
            self.chat_model._build_chat_params(messages)

            # Check that ChatRequest was called with the right parameters
            mock_chat_request.assert_called_once()

            # Test setting additional properties
            self.chat_model.chat_id = "test-chat-id"
            request = self.chat_model._build_chat_params(messages)
            assert request.chat_id == "test-chat-id"

            # Test inclusions
            self.chat_model.inclusions = {"datasources": ["confluence", "drive"]}
            with patch("langchain_glean.chat_models.chat.models.ChatRestrictionFilters", autospec=True) as mock_filters:
                mock_filters_instance = MagicMock()
                mock_filters.return_value = mock_filters_instance

                # Reset mock_chat_request to avoid previous calls affecting our test
                mock_chat_request.reset_mock()

                request = self.chat_model._build_chat_params(messages)
                mock_chat_request.assert_called_once()
                assert hasattr(request, "inclusions")

            # Test exclusions
            self.chat_model.exclusions = {"datasources": ["slack"]}
            with patch("langchain_glean.chat_models.chat.models.ChatRestrictionFilters", autospec=True) as mock_filters:
                mock_filters_instance = MagicMock()
                mock_filters.return_value = mock_filters_instance

                # Reset mock_chat_request to avoid previous calls affecting our test
                mock_chat_request.reset_mock()

                request = self.chat_model._build_chat_params(messages)
                mock_chat_request.assert_called_once()
                assert hasattr(request, "exclusions")

            # Test timeout_millis
            self.chat_model.timeout_millis = 30000
            request = self.chat_model._build_chat_params(messages)
            assert request.timeout_millis == 30000

            # Test application_id
            self.chat_model.application_id = "custom-app"
            request = self.chat_model._build_chat_params(messages)
            assert request.application_id == "custom-app"

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
        self.chat_model._client.chat.create.assert_called_once()
