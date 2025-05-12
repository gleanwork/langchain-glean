from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from glean import Glean, errors, models
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, model_validator


class ChatGlean(BaseChatModel):
    """`Glean` Chat large language model API.

    To use, you should have the environment variables ``GLEAN_API_TOKEN`` and
    ``GLEAN_SUBDOMAIN`` set with your API token and Glean subdomain. If using a global token,
    you should also set ``GLEAN_ACT_AS`` with the email of the user to act as.

    Setup:
        Install ``langchain-glean`` and set the required environment variables.

        .. code-block:: bash

            pip install -U langchain-glean
            export GLEAN_API_TOKEN="your-api-token"  # Can be a global or user token
            export GLEAN_SUBDOMAIN="your-glean-subdomain"
            export GLEAN_ACT_AS="user@example.com"  # Only required for global tokens

    Key init args:
        api_token: Optional[str]
            Glean API token. If not provided, will be read from GLEAN_API_TOKEN env var.
        subdomain: Optional[str]
            Glean subdomain. If not provided, will be read from GLEAN_SUBDOMAIN env var.
        act_as: Optional[str]
            Email of the user to act as when using a global token. If not provided,
            will be read from GLEAN_ACT_AS env var.
        chat_id: Optional[str]
            ID of an existing chat to continue. If not provided, a new chat will be created.
        save_chat: bool
            Whether to save the chat session for future use. Default is False.
        agent_config: Dict[str, Any]
            Configuration for the agent that will execute the request. Contains 'agent' and 'mode' parameters.
            Default is {"agent": "DEFAULT", "mode": "DEFAULT"}.
        timeout: Optional[int]
            Timeout for API requests in seconds. Default is 60.
        inclusions: Optional[Dict[str, Any]]
            A list of filters which only allows chat to access certain content.
        exclusions: Optional[Dict[str, Any]]
            A list of filters which disallows chat from accessing certain content.
            If content is in both inclusions and exclusions, it'll be excluded.
        timeout_millis: Optional[int]
            Timeout in milliseconds for the request. A 408 error will be returned if
            handling the request takes longer.
        application_id: Optional[str]
            The ID of the application this request originates from, used to determine
            the configuration of underlying chat processes.
        model_kwargs: Dict[str, Any]
            Additional parameters to pass to the chat API.

    Instantiate:
        .. code-block:: python

            from langchain_glean.chat_models import ChatGlean

            # Using environment variables
            chat = ChatGlean()

            # Or explicitly providing credentials
            chat = ChatGlean(
                api_token="your-api-token",
                subdomain="your-glean-subdomain",
                act_as="user@example.com",  # Only required for global tokens
                save_chat=True,
                timeout=60,
            )

            # Using advanced parameters
            chat = ChatGlean(
                api_token="your-api-token",
                subdomain="your-glean-subdomain",
                agent_config={"agent": "GPT", "mode": "SEARCH"},  # Configure agent and mode
                inclusions={"datasources": ["confluence", "drive"]},  # Only search in these datasources
                exclusions={"datasources": ["slack"]},  # Exclude these datasources
                timeout_millis=30000,  # 30 seconds server-side timeout
                application_id="custom-app",  # Custom application ID
            )

    Invoke:
        .. code-block:: python

            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(content="What are the company holidays this year?")
            ]

            response = chat.invoke(messages)
            print(response.content)
    """

    instance: str = Field(description="Instance for Glean")
    api_token: str = Field(description="API token for Glean")
    act_as: Optional[str] = Field(
        default=None, description="Email for the user to act as. Required only when using a global token, not needed for user tokens."
    )
    save_chat: bool = Field(default=False, description="Whether to save the chat session in Glean.")
    chat_id: Optional[str] = Field(default=None, description="ID of an existing chat to continue. If None, a new chat will be created.")
    timeout: int = Field(default=60, description="Timeout in seconds for the API request.")

    # Agent configuration
    agent_config: Dict[str, Any] = Field(
        default_factory=lambda: {"agent": "DEFAULT", "mode": "DEFAULT"},
        description="Configuration for the agent that will execute the request. Contains 'agent' and 'mode' parameters.",
    )

    # Additional parameters from the OpenAPI specification
    inclusions: Optional[Dict[str, Any]] = Field(default=None, description="A list of filters which only allows chat to access certain content.")
    exclusions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A list of filters which disallows chat from accessing certain content. "
        "If content is in both inclusions and exclusions, it'll be excluded.",
    )
    timeout_millis: Optional[int] = Field(
        default=None, description="Timeout in milliseconds for the request. A 408 error will be returned if handling the request takes longer."
    )
    application_id: Optional[str] = Field(
        default=None, description="The ID of the application this request originates from, used to determine the configuration of underlying chat processes."
    )

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and subdomain exists in environment.

        Args:
            values: The values to validate.

        Returns:
            The validated values.

        Raises:
            ValueError: If api key or subdomain are not found in environment.
        """
        values = values or {}
        values["instance"] = get_from_dict_or_env(values, "instance", "GLEAN_INSTANCE")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")

        return values

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chat model.

        Args:
            **kwargs: Keyword arguments to pass to the parent class.
        """
        super().__init__(**kwargs)

        try:
            g = Glean(api_token=self.api_token, instance=self.instance)
            self._client = g.client
        except Exception as e:
            raise ValueError(f"Failed to initialize Glean client: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "glean-chat"

    def _convert_message_to_glean_format(self, message: BaseMessage) -> models.ChatMessage:
        """Convert a LangChain message to Glean's message format.

        Args:
            message: The LangChain message to convert.

        Returns:
            The message in Glean's format.
        """
        if isinstance(message, HumanMessage):
            author = models.Author.USER
        elif isinstance(message, AIMessage):
            author = models.Author.GLEAN_AI
        elif isinstance(message, SystemMessage):
            # System messages are treated as context messages in Glean
            author = models.Author.USER
            return models.ChatMessage(author=author, message_type=models.MessageType.CONTEXT, fragments=[models.ChatMessageFragment(text=str(message.content))])
        elif isinstance(message, ChatMessage):
            # Map custom roles to Glean's format
            if message.role.upper() == "USER":
                author = models.Author.USER
            elif message.role.upper() == "ASSISTANT" or message.role.upper() == "AI":
                author = models.Author.GLEAN_AI
            else:
                # Default to USER for unknown roles
                author = models.Author.USER
        else:
            # Default to USER for unknown message types
            author = models.Author.USER

        return models.ChatMessage(author=author, message_type=models.MessageType.CONTENT, fragments=[models.ChatMessageFragment(text=str(message.content))])

    def _convert_glean_message_to_langchain(self, message: models.ChatMessage) -> BaseMessage:
        """Convert a Glean message to a LangChain message.

        Args:
            message: The Glean message to convert.

        Returns:
            The message in LangChain's format.
        """
        author = message.author if hasattr(message, "author") else None
        fragments = message.fragments if hasattr(message, "fragments") else []

        content = ""
        if fragments:
            for fragment in fragments:
                if hasattr(fragment, "text") and fragment.text:
                    content += fragment.text

        if author == models.Author.GLEAN_AI:
            return AIMessage(content=content)
        else:
            return HumanMessage(content=content)

    def _build_chat_params(self, messages: List[BaseMessage]) -> models.ChatRequest:
        """Create a chat request for the Glean API.

        Args:
            messages: The messages to include in the request.

        Returns:
            The chat request in Glean's format.
        """
        glean_messages = [self._convert_message_to_glean_format(msg) for msg in messages]

        # Convert agent_config to GleanAgentConfig
        agent_config = None
        if self.agent_config:
            agent_config = models.AgentConfig(agent=self.agent_config.get("agent", "DEFAULT"), mode=self.agent_config.get("mode", "DEFAULT"))

        # Build ChatRequest with required parameters
        request = models.ChatRequest(messages=glean_messages, save_chat=self.save_chat, agent_config=agent_config)

        # Add optional parameters if they are set
        if self.chat_id:
            request.chat_id = self.chat_id

        # Convert inclusions/exclusions to proper type if needed
        if self.inclusions:
            inclusions = models.ChatRestrictionFilters(**self.inclusions) if isinstance(self.inclusions, dict) else self.inclusions
            request.inclusions = inclusions

        if self.exclusions:
            exclusions = models.ChatRestrictionFilters(**self.exclusions) if isinstance(self.exclusions, dict) else self.exclusions
            request.exclusions = exclusions

        if self.timeout_millis:
            request.timeout_millis = self.timeout_millis

        if self.application_id:
            request.application_id = self.application_id

        return request

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from Glean.

        Args:
            messages: The messages to generate a response for.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for the run.
            **kwargs: Additional keyword arguments.

        Returns:
            A ChatResult containing the generated response.

        Raises:
            ValueError: If the response from Glean is invalid.
        """
        if stop is not None:
            raise ValueError("stop sequences are not supported by the Glean Chat Model")

        params = self._build_chat_params(messages)

        try:
            # Call chat.create with the chat request
            response = self._client.chat.create(
                messages=params.messages,
                save_chat=params.save_chat,
                chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                agent_config=params.agent_config,
                inclusions=params.inclusions,
                exclusions=params.exclusions,
                timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                application_id=params.application_id if hasattr(params, "application_id") else None,
            )

        except errors.GleanError as client_err:
            raise ValueError(f"Glean client error: {str(client_err)}")

        # Extract AI messages from the response
        ai_messages = []
        if response and response.messages:
            ai_messages = [
                msg for msg in response.messages if isinstance(msg, dict) and msg.get("author") == "GLEAN_AI" and msg.get("messageType") == "CONTENT"
            ]

        if not ai_messages:
            raise ValueError("No AI response found in the Glean response")

        ai_message = ai_messages[-1]

        # Use proper attribute access for ChatResponse
        if hasattr(response, "chatId") and response.chatId:
            self.chat_id = response.chatId

        langchain_message = self._convert_glean_message_to_langchain(ai_message)

        # Create the generation with metadata
        generation = ChatGeneration(
            message=langchain_message,
            generation_info={
                "chat_id": self.chat_id,
                "tracking_token": response.chatSessionTrackingToken if hasattr(response, "chatSessionTrackingToken") else None,
            },
        )

        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from Glean asynchronously.

        Args:
            messages: The messages to generate a response for.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for the run.
            **kwargs: Additional keyword arguments.

        Returns:
            A ChatResult containing the generated response.

        Raises:
            ValueError: If the response from Glean is invalid.
        """
        if stop is not None:
            raise ValueError("stop sequences are not supported by the Glean Chat Model")

        params = self._build_chat_params(messages)

        try:
            # Call chat.create_async with the chat request
            response = await self._client.chat.create_async(
                messages=params.messages,
                save_chat=params.save_chat,
                chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                agent_config=params.agent_config,
                inclusions=params.inclusions,
                exclusions=params.exclusions,
                timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                application_id=params.application_id if hasattr(params, "application_id") else None,
            )

        except errors.GleanError as client_err:
            raise ValueError(f"Glean client error: {str(client_err)}")

        # Extract AI messages from the response
        ai_messages = []
        if response and response.messages:
            ai_messages = [
                msg for msg in response.messages if isinstance(msg, dict) and msg.get("author") == "GLEAN_AI" and msg.get("messageType") == "CONTENT"
            ]

        if not ai_messages:
            raise ValueError("No AI response found in the Glean response")

        ai_message = ai_messages[-1]

        # Use proper attribute access for ChatResponse
        if hasattr(response, "chatId") and response.chatId:
            self.chat_id = response.chatId

        langchain_message = self._convert_glean_message_to_langchain(ai_message)

        # Create the generation with metadata
        generation = ChatGeneration(
            message=langchain_message,
            generation_info={
                "chat_id": self.chat_id,
                "tracking_token": response.chatSessionTrackingToken if hasattr(response, "chatSessionTrackingToken") else None,
            },
        )

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a chat response from Glean.

        Args:
            messages: The messages to generate a response for.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for the run.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk: Chunks of the generated chat response.

        Raises:
            ValueError: If there's an error with the Glean API call or response processing.
        """
        if stop is not None:
            raise ValueError("stop sequences are not supported by the Glean Chat Model")

        params = self._build_chat_params(messages)
        params.stream = True

        try:
            # Use the chat streaming endpoint with proper parameters
            response_stream = self._client.chat.create_stream(
                messages=params.messages,
                save_chat=params.save_chat,
                chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                agent_config=params.agent_config,
                inclusions=params.inclusions,
                exclusions=params.exclusions,
                timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                application_id=params.application_id if hasattr(params, "application_id") else None,
                stream=True,
            )

            # Parse the response stream line by line
            for line in response_stream.splitlines():
                if not line.strip():
                    continue

                try:
                    # Parse the JSON line into a ChatResponse object
                    import json

                    chunk_data = json.loads(line)
                    if "messages" in chunk_data:
                        for message in chunk_data["messages"]:
                            if isinstance(message, dict) and message.get("author") == "GLEAN_AI" and message.get("messageType") == "CONTENT":
                                for fragment in message.get("fragments", []):
                                    if "text" in fragment:
                                        new_content = fragment.get("text", "")
                                        if new_content:
                                            message_chunk = AIMessageChunk(content=new_content)

                                            chat_id = chunk_data.get("chatId")
                                            if chat_id and not self.chat_id:
                                                self.chat_id = chat_id

                                            tracking_token = chunk_data.get("chatSessionTrackingToken")

                                            gen_chunk = ChatGenerationChunk(
                                                message=message_chunk, generation_info={"chat_id": chat_id, "tracking_token": tracking_token}
                                            )
                                            yield gen_chunk

                                            if run_manager:
                                                run_manager.on_llm_new_token(new_content)
                except Exception as parsing_error:
                    if run_manager:
                        run_manager.on_llm_error(parsing_error)
                    raise ValueError(f"Error parsing stream response: {str(parsing_error)}")

        except errors.GleanError as client_err:
            if run_manager:
                run_manager.on_llm_error(client_err)
            raise ValueError(f"Glean client error: {str(client_err)}")
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise ValueError(f"Error during streaming: {str(e)}")

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream a chat response from Glean asynchronously.

        Args:
            messages: The messages to generate a response for.
            stop: A list of strings to stop generation when encountered.
            run_manager: A callback manager for the run.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk: Chunks of the generated chat response.

        Raises:
            ValueError: If there's an error with the Glean API call or response processing.
        """
        if stop is not None:
            raise ValueError("stop sequences are not supported by the Glean Chat Model")

        params = self._build_chat_params(messages)
        params.stream = True

        try:
            # Use the async chat streaming endpoint with proper parameters
            response_stream = await self._client.chat.create_stream_async(
                messages=params.messages,
                save_chat=params.save_chat,
                chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                agent_config=params.agent_config,
                inclusions=params.inclusions,
                exclusions=params.exclusions,
                timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                application_id=params.application_id if hasattr(params, "application_id") else None,
                stream=True,
            )

            # Parse the response stream line by line
            for line in response_stream.splitlines():
                if not line.strip():
                    continue

                try:
                    # Parse the JSON line into a ChatResponse object
                    import json

                    chunk_data = json.loads(line)
                    if "messages" in chunk_data:
                        for message in chunk_data["messages"]:
                            if isinstance(message, dict) and message.get("author") == "GLEAN_AI" and message.get("messageType") == "CONTENT":
                                for fragment in message.get("fragments", []):
                                    if "text" in fragment:
                                        new_content = fragment.get("text", "")
                                        if new_content:
                                            message_chunk = AIMessageChunk(content=new_content)

                                            chat_id = chunk_data.get("chatId")
                                            if chat_id and not self.chat_id:
                                                self.chat_id = chat_id

                                            tracking_token = chunk_data.get("chatSessionTrackingToken")

                                            gen_chunk = ChatGenerationChunk(
                                                message=message_chunk, generation_info={"chat_id": chat_id, "tracking_token": tracking_token}
                                            )
                                            yield gen_chunk

                                            if run_manager:
                                                await run_manager.on_llm_new_token(new_content)
                except Exception as parsing_error:
                    if run_manager:
                        await run_manager.on_llm_error(parsing_error)
                    raise ValueError(f"Error parsing stream response: {str(parsing_error)}")

        except errors.GleanError as client_err:
            if run_manager:
                await run_manager.on_llm_error(client_err)
            raise ValueError(f"Glean client error: {str(client_err)}")
        except Exception as e:
            if run_manager:
                await run_manager.on_llm_error(e)
            raise ValueError(f"Error during streaming: {str(e)}")
