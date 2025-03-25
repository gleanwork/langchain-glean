import json
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
        os.environ["GLEAN_SUBDOMAIN"] = "test-subdomain"
        os.environ["GLEAN_API_TOKEN"] = "test-api-token"

        self.mock_client_patcher = patch("langchain_glean.chat_models.chat.GleanClient")
        self.mock_client = self.mock_client_patcher.start()

        self.mock_auth_patcher = patch("langchain_glean.chat_models.chat.GleanAuth")
        self.mock_auth = self.mock_auth_patcher.start()

        self.field_patcher = patch("langchain_glean.chat_models.chat.Field", side_effect=lambda default=None, **kwargs: default)
        self.field_mock = self.field_patcher.start()

        self.chat_model = ChatGlean()

        self.chat_model._client = MagicMock()
        mock_response = MagicMock()
        self.chat_model._client.post.return_value = mock_response
        self.chat_model._client.parse_response.return_value = self._get_mock_response()

        self.chat_model._client.session = MagicMock()
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = self._get_mock_stream_lines()
        self.chat_model._client.session.post.return_value = mock_response

        self.chat_model._client.base_url = "https://example.glean.com/api/v1"

        yield

        # Clean up after tests
        self.mock_client_patcher.stop()
        self.mock_auth_patcher.stop()
        self.field_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def _get_mock_response(self):
        """Get a mock response from the Glean chat API."""
        return {
            "chatId": "mock-chat-id",
            "chatSessionTrackingToken": "mock-tracking-token",
            "messages": [{"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "This is a mock response from Glean AI."}]}],
        }

    def _get_mock_stream_lines(self):
        """Get mock stream lines for the streaming response."""
        data1 = json.dumps(
            {
                "chatId": "mock-chat-id",
                "chatSessionTrackingToken": "mock-tracking-token",
            }
        )
        line1 = f"data: {data1}".encode("utf-8")

        data2 = json.dumps({"messages": [{"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "This is "}]}]})
        line2 = f"data: {data2}".encode("utf-8")

        data3 = json.dumps({"messages": [{"author": "GLEAN_AI", "messageType": "CONTENT", "fragments": [{"text": "a streaming response."}]}]})
        line3 = f"data: {data3}".encode("utf-8")

        return [line1, line2, line3]

    def test_initialization(self):
        """Test that the chat model initializes correctly."""
        assert self.chat_model is not None
        assert hasattr(self.chat_model, "_client")
        assert hasattr(self.chat_model, "_auth")

    def test_convert_message_to_glean_format(self):
        """Test converting LangChain messages to Glean format."""
        human_msg = HumanMessage(content="Hello, Glean!")
        glean_msg = self.chat_model._convert_message_to_glean_format(human_msg)
        assert glean_msg["author"] == "USER"
        assert glean_msg["messageType"] == "CONTENT"
        assert glean_msg["fragments"][0]["text"] == "Hello, Glean!"

        ai_msg = AIMessage(content="Hello, human!")
        glean_msg = self.chat_model._convert_message_to_glean_format(ai_msg)
        assert glean_msg["author"] == "GLEAN_AI"
        assert glean_msg["messageType"] == "CONTENT"
        assert glean_msg["fragments"][0]["text"] == "Hello, human!"

        system_msg = SystemMessage(content="You are an AI assistant.")
        glean_msg = self.chat_model._convert_message_to_glean_format(system_msg)
        assert glean_msg["author"] == "USER"
        assert glean_msg["messageType"] == "CONTEXT"
        assert glean_msg["fragments"][0]["text"] == "You are an AI assistant."

    def test_create_chat_request(self):
        """Test creating a chat request from messages."""
        messages = [SystemMessage(content="You are a helpful AI assistant."), HumanMessage(content="Hello, Glean!")]

        self.chat_model.save_chat = False

        request = self.chat_model._build_chat_params(messages)

        assert "messages" in request
        assert len(request["messages"]) == 2
        assert request["messages"][0]["author"] == "USER"
        assert request["messages"][0]["messageType"] == "CONTEXT"
        assert request["messages"][1]["author"] == "USER"
        assert "saveChat" in request
        assert request["saveChat"] is False

        self.chat_model.save_chat = True
        request = self.chat_model._build_chat_params(messages)
        assert request["saveChat"] is True

        self.chat_model.chat_id = "test-chat-id"
        request = self.chat_model._build_chat_params(messages)
        assert request["chatId"] == "test-chat-id"

        # Test agent_config parameter
        self.chat_model.agent_config = {"agent": "GPT", "mode": "DEFAULT"}
        request = self.chat_model._build_chat_params(messages)
        assert "agentConfig" in request
        assert "agent" in request["agentConfig"]
        assert request["agentConfig"]["agent"] == "GPT"
        assert "mode" in request["agentConfig"]
        assert request["agentConfig"]["mode"] == "DEFAULT"

        # Update agent_config with different mode
        self.chat_model.agent_config = {"agent": "GPT", "mode": "SEARCH"}
        request = self.chat_model._build_chat_params(messages)
        assert "agentConfig" in request
        assert "agent" in request["agentConfig"]
        assert request["agentConfig"]["agent"] == "GPT"
        assert "mode" in request["agentConfig"]
        assert request["agentConfig"]["mode"] == "SEARCH"

        # Test new parameters
        self.chat_model.inclusions = {"datasources": ["confluence", "drive"]}
        request = self.chat_model._build_chat_params(messages)
        assert "inclusions" in request
        assert request["inclusions"] == {"datasources": ["confluence", "drive"]}

        self.chat_model.exclusions = {"datasources": ["slack"]}
        request = self.chat_model._build_chat_params(messages)
        assert "exclusions" in request
        assert request["exclusions"] == {"datasources": ["slack"]}

        self.chat_model.timeout_millis = 30000
        request = self.chat_model._build_chat_params(messages)
        assert "timeoutMillis" in request
        assert request["timeoutMillis"] == 30000

        self.chat_model.application_id = "custom-app"
        request = self.chat_model._build_chat_params(messages)
        assert "applicationId" in request
        assert request["applicationId"] == "custom-app"

    def test_generate(self):
        """Test generating a response from the chat model."""

        with patch.object(self.chat_model, "_build_chat_params") as mock_build_params:
            mock_build_params.return_value = {
                "messages": [
                    {"author": "USER", "messageType": "CONTEXT", "fragments": [{"text": "You are a helpful AI assistant."}]},
                    {"author": "USER", "messageType": "CONTENT", "fragments": [{"text": "Hello, Glean!"}]},
                ],
                "saveChat": False,
                "agentConfig": {"agent": "DEFAULT", "mode": "DEFAULT"},
            }

            result = self.chat_model._generate(self.messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "This is a mock response from Glean AI."
        assert result.generations[0].generation_info["chat_id"] == "mock-chat-id"
        assert result.generations[0].generation_info["tracking_token"] == "mock-tracking-token"

        assert self.chat_model.chat_id == "mock-chat-id"

        self.chat_model._client.post.assert_called_once()
        args, kwargs = self.chat_model._client.post.call_args
        assert args[0] == "chat"
        assert "data" in kwargs
        assert "headers" in kwargs
        assert "timeout" in kwargs
