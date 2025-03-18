import os
from typing import Any, Dict, List, Type

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_glean.chat_models import ChatGlean


class TestChatGlean(ChatModelIntegrationTests):
    """Test the ChatGlean with actual API calls."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_SUBDOMAIN") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def model_class(self) -> Type[BaseChatModel]:
        """Return the model class to test."""
        return ChatGlean

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """Return model kwargs to use for testing."""
        return {}

    @property
    def model_integration_kwargs(self) -> Dict[str, Any]:
        """Return model kwargs to use for integration testing."""
        return {}

    @property
    def messages(self) -> List[BaseMessage]:
        """Return messages to use for testing."""
        return [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Give a short greeting."),
        ]

    @property
    def messages_with_system(self) -> List[BaseMessage]:
        """Return messages with a system message to use for testing."""
        return [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Give a short greeting."),
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

    def setup_method(self):
        """Set up the test."""
        self.chat_model = ChatGlean()
        self.messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Give a short greeting."),
        ]

    def test_invoke(self):
        """Test the invoke method."""
        response = self.chat_model.invoke(self.messages)

        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    def test_batch(self):
        """Test the batch method."""
        batch_messages = [
            self.messages,
            [HumanMessage(content="What is today's date?")],
        ]

        responses = self.chat_model.batch(batch_messages)

        assert len(responses) == 2
        for response in responses:
            assert isinstance(response, AIMessage)
            assert isinstance(response.content, str)
            assert len(response.content) > 0

    def test_generate(self):
        """Test the generate method."""
        result = self.chat_model.generate([self.messages])

        assert isinstance(result, LLMResult)
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        assert isinstance(result.generations[0][0].message, AIMessage)
        assert isinstance(result.generations[0][0].message.content, str)
        assert len(result.generations[0][0].message.content) > 0

    def test_stream(self):
        """Test the stream method."""
        chunks = list(self.chat_model.stream(self.messages))

        assert len(chunks) > 0
        for chunk in chunks:
            assert hasattr(chunk, "message")
            assert hasattr(chunk.message, "content")

    def test_with_chat_history(self):
        """Test using the chat model with chat history."""
        chat_model = ChatGlean(save_chat=True)

        conversation: List[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="What is the capital of France?"),
        ]

        response1 = chat_model.invoke(conversation)
        conversation.append(response1)

        assert isinstance(response1, AIMessage)
        assert "Paris" in response1.content

        assert chat_model.chat_id is not None
        chat_id = chat_model.chat_id

        conversation.append(HumanMessage(content="What is its population?"))
        response2 = chat_model.invoke(conversation)

        assert isinstance(response2, AIMessage)
        assert len(response2.content) > 0

        assert chat_model.chat_id == chat_id

    def test_with_different_parameters(self):
        """Test the chat model with different parameters."""
        chat_model = ChatGlean(agent="GPT")
        response = chat_model.invoke(self.messages)
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        chat_model = ChatGlean(mode="SEARCH")
        response = chat_model.invoke(self.messages)
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        chat_model = ChatGlean(timeout=30)
        response = chat_model.invoke(self.messages)
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

    def test_continue_existing_chat(self):
        """Test continuing an existing chat."""
        chat_model1 = ChatGlean(save_chat=True)
        chat_response = chat_model1.invoke(self.messages)
        assert chat_response.content
        chat_id = chat_model1.chat_id

        chat_model2 = ChatGlean(chat_id=chat_id)

        follow_up = [HumanMessage(content="Tell me more.")]
        response2 = chat_model2.invoke(follow_up)

        assert isinstance(response2, AIMessage)
        assert len(response2.content) > 0
