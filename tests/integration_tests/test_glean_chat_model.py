import os
import unittest
from typing import List, Type

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_glean.chat_models import ChatGlean
from langchain_glean.chat_models.chat import ChatBasicRequest


class TestGleanChatModelIntegration(unittest.TestCase):
    """Integration tests for the ChatGlean model."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def model_class(self) -> Type[BaseChatModel]:
        """Get the chat model class."""
        return ChatGlean

    @property
    def model_params(self) -> dict:
        """Get parameters for initializing the model."""
        return {}  # No params needed as we use environment variables

    @property
    def basic_messages(self) -> List[BaseMessage]:
        """Return basic messages for testing."""
        return [HumanMessage(content="What can Glean's assistant do?")]

    @property
    def messages_with_system(self) -> List[BaseMessage]:
        """Return messages with a system message for testing."""
        return [SystemMessage(content="You are a helpful assistant that provides concise responses."), HumanMessage(content="What can Glean's assistant do?")]

    @property
    def messages_with_history(self) -> List[BaseMessage]:
        """Return messages with chat history for testing."""
        return [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What features does Glean offer?"),
            AIMessage(content="Glean offers enterprise search, AI assistant capabilities, and more."),
            HumanMessage(content="Can you elaborate on the AI assistant capabilities?"),
        ]

    def test_invoke_with_basic_messages(self) -> None:
        """Test invoking with basic messages."""
        chat = self.model_class(**self.model_params)
        response = chat.invoke(self.basic_messages)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    def test_invoke_with_system_messages(self) -> None:
        """Test invoking with system messages."""
        chat = self.model_class(**self.model_params)
        response = chat.invoke(self.messages_with_system)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    def test_invoke_with_chat_history(self) -> None:
        """Test invoking with chat history."""
        chat = self.model_class(**self.model_params)
        response = chat.invoke(self.messages_with_history)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    def test_stream(self) -> None:
        """Test streaming responses."""
        chat = self.model_class(**self.model_params)

        response_chunks = []
        for chunk in chat.stream(self.basic_messages):
            self.assertIsInstance(chunk, AIMessage)
            response_chunks.append(chunk.content)

        self.assertTrue(len(response_chunks) > 0)

        full_response = "".join(response_chunks)
        self.assertTrue(len(full_response) > 0)

    def test_invoke_with_basic_request(self) -> None:
        """Test invoking with a ChatBasicRequest."""
        chat = self.model_class(**self.model_params)

        request = ChatBasicRequest(message="What can Glean's assistant do?")

        response = chat.invoke(request)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    def test_invoke_with_request_and_context(self) -> None:
        """Test invoking with a request that includes context."""
        chat = self.model_class(**self.model_params)

        request = ChatBasicRequest(
            message="Summarize this information",
            context=["Glean is an enterprise search platform.", "It uses AI to provide better search results.", "It includes tools for knowledge management."],
        )

        response = chat.invoke(request)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    def test_invoke_with_agent_config(self) -> None:
        """Test invoking with agent configuration."""
        chat = self.model_class(**self.model_params)

        request = ChatBasicRequest(message="Give me a short answer to: What is Glean?")

        response = chat.invoke(request, agent="GPT", mode="QUICK")

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    async def test_ainvoke_with_basic_request(self) -> None:
        """Test async invocation with a basic request."""
        chat = self.model_class(**self.model_params)

        request = ChatBasicRequest(message="What can Glean's assistant do?")

        response = await chat.ainvoke(request)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)
