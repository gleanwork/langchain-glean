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

from langchain_glean.chat_models import ChatGleanAgent


class TestGleanAgentChatModelIntegration(unittest.TestCase):
    """Integration tests for the ChatGleanAgent model."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

        # Define a test agent ID - this should be a valid agent ID in your Glean instance
        self.agent_id = os.environ.get("TEST_AGENT_ID")
        if not self.agent_id:
            self.skipTest("TEST_AGENT_ID not found in environment variables")

    @property
    def model_class(self) -> Type[BaseChatModel]:
        """Get the chat model class."""
        return ChatGleanAgent

    @property
    def model_params(self) -> dict:
        """Get parameters for initializing the model."""
        return {"agent_id": self.agent_id}

    @property
    def basic_messages(self) -> List[BaseMessage]:
        """Return basic messages for testing."""
        return [HumanMessage(content="What can this agent do?")]

    @property
    def messages_with_system(self) -> List[BaseMessage]:
        """Return messages with a system message for testing."""
        return [SystemMessage(content="You are a helpful assistant."), HumanMessage(content="What can this agent do?")]

    @property
    def messages_with_history(self) -> List[BaseMessage]:
        """Return messages with chat history for testing."""
        return [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What features does Glean offer?"),
            AIMessage(content="Glean offers enterprise search, AI assistant capabilities, and more."),
            HumanMessage(content="Can you elaborate on agent capabilities?"),
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

    def test_invoke_with_custom_fields(self) -> None:
        """Test invoking with custom fields."""
        chat = self.model_class(**self.model_params)

        # Add custom fields based on the agent's schema
        # This is just an example; adjust according to your agent's schema
        custom_fields = {"input": "Test input", "param1": "value1"}

        response = chat.invoke(self.basic_messages, fields=custom_fields)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    async def test_ainvoke_with_basic_messages(self) -> None:
        """Test async invoking with basic messages."""
        chat = self.model_class(**self.model_params)
        response = await chat.ainvoke(self.basic_messages)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)

    async def test_ainvoke_with_custom_fields(self) -> None:
        """Test async invoking with custom fields."""
        chat = self.model_class(**self.model_params)

        # Add custom fields based on the agent's schema
        # This is just an example; adjust according to your agent's schema
        custom_fields = {"input": "Test input", "param1": "value1"}

        response = await chat.ainvoke(self.basic_messages, fields=custom_fields)

        self.assertIsInstance(response, AIMessage)
        self.assertTrue(len(response.content) > 0)
