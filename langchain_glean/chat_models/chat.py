from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast

from glean.api_client import Glean, errors, models  # noqa: F401
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
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
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class ChatBasicRequest(BaseModel):
    """Basic chat request: a single user message plus optional context messages."""

    message: str = Field(..., description="The user question or statement.")
    context: Optional[List[str]] = Field(
        default=None,
        description="Optional list of earlier user utterances – ordered oldest → newest.",
    )


class ChatGlean(GleanAPIClientMixin, BaseChatModel):
    """`Glean` Chat model wrapper.

    Setup
    -----
    Install *langchain-glean* and export the required credentials::

        .. code-block:: bash

            pip install -U langchain-glean
            export GLEAN_API_TOKEN="your-api-token"   # user or global token
            export GLEAN_INSTANCE="acme"              # your Glean sub-domain
            export GLEAN_ACT_AS="user@example.com"    # only for global tokens

    Key init args
    -------------
    api_token : str, optional
        Glean API token.  Falls back to ``GLEAN_API_TOKEN``.
    instance : str, optional
        Glean instance / sub-domain (``GLEAN_INSTANCE``).
    act_as : str, optional
        Email to impersonate when using a global token (``GLEAN_ACT_AS``).
    chat_id : str, optional
        Continue an existing chat session or inspect the ID after the first call via the :pyattr:`chat_id` property.
    model_kwargs : Dict[str, Any]
        Extra parameters forwarded to the underlying Glean client.

    Per-call overrides
    ------------------
    Chat-specific knobs such as ``save_chat``, ``agent_config``, ``timeout_millis``,
    ``inclusions``/``exclusions`` *must* be supplied **per invocation** via
    :py:meth:`invoke`, :py:meth:`ainvoke`, :py:meth:`stream`, or by passing a fully
    populated :class:`glean.models.ChatRequest`.

    Instantiate
    -----------
    .. code-block:: python

        from langchain_glean.chat_models import ChatGlean

        chat = ChatGlean()                      # reads env-vars
        chat = ChatGlean(api_token="token", instance="acme")

    Invoke
    ------
    .. code-block:: python

        from langchain_core.messages import HumanMessage
        from langchain_glean.chat_models import ChatGlean

        chat = ChatGlean()
        response = chat.invoke([HumanMessage(content="Hello")])
        print(response.content)
    """

    _chat_id: Optional[str] = PrivateAttr(default=None)
    model_config = ConfigDict(extra="allow")

    @property
    def chat_id(self) -> Optional[str]:  # noqa: D401
        """ID of the current chat session.

        After the first successful :py:meth:`invoke` the value is populated from
        the API response.  Set it manually to resume an existing conversation::

            chat = ChatGlean()
            chat.chat_id = "abc123"
            chat.invoke([...])
        """

        return self._chat_id

    @chat_id.setter
    def chat_id(self, value: Optional[str]) -> None:  # noqa: D401
        self._chat_id = value

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
            author = models.Author.USER
            return models.ChatMessage(author=author, message_type=models.MessageType.CONTEXT, fragments=[models.ChatMessageFragment(text=str(message.content))])
        elif isinstance(message, ChatMessage):
            if message.role.upper() == "USER":
                author = models.Author.USER
            elif message.role.upper() == "ASSISTANT" or message.role.upper() == "AI":
                author = models.Author.GLEAN_AI
            else:
                author = models.Author.USER
        else:
            author = models.Author.USER

        return models.ChatMessage(author=author, message_type=models.MessageType.CONTENT, fragments=[models.ChatMessageFragment(text=str(message.content))])

    def _convert_glean_message_to_langchain(self, message: models.ChatMessage) -> BaseMessage:
        """Convert a Glean message to a LangChain message.

        Args:
            message: The Glean message to convert.

        Returns:
            The message in LangChain's format.
        """
        content = ""
        if message.fragments:
            for fragment in message.fragments:
                if fragment.text:
                    content += fragment.text

        if message.author == models.Author.GLEAN_AI:
            return AIMessage(content=content)
        else:
            return HumanMessage(content=content)

    def _build_chat_params(self, messages: List[BaseMessage], **overrides: Any) -> models.ChatRequest:
        """Create a chat request for the Glean API.

        Args:
            messages: The messages to include in the request.
            **overrides: Additional keyword arguments to override default parameters.

        Returns:
            The chat request in Glean's format.
        """
        glean_messages = [self._convert_message_to_glean_format(msg) for msg in messages]

        agent_config_arg = overrides.get("agent_config")
        if agent_config_arg is None:
            agent_config_arg = {"agent": "DEFAULT", "mode": "DEFAULT"}

        agent_config: Optional[models.AgentConfig] = None
        if agent_config_arg is not None:
            if isinstance(agent_config_arg, dict):
                agent_config = models.AgentConfig(agent=agent_config_arg.get("agent", "DEFAULT"), mode=agent_config_arg.get("mode", "DEFAULT"))
            else:
                agent_config = agent_config_arg  # type: ignore[assignment]

        save_chat_flag = bool(overrides.get("save_chat", False))

        request = models.ChatRequest(messages=glean_messages, save_chat=save_chat_flag, agent_config=agent_config)

        chat_id_val = overrides.get("chat_id", self._chat_id)
        if chat_id_val:
            request.chat_id = chat_id_val

        inclusions_val = overrides.get("inclusions")
        exclusions_val = overrides.get("exclusions")

        if inclusions_val is not None:
            if isinstance(inclusions_val, dict):
                inclusions_val = models.ChatRestrictionFilters(**inclusions_val)
            request.inclusions = inclusions_val  # type: ignore[assignment]

        if exclusions_val is not None:
            if isinstance(exclusions_val, dict):
                exclusions_val = models.ChatRestrictionFilters(**exclusions_val)
            request.exclusions = exclusions_val  # type: ignore[assignment]

        timeout_ms_val = overrides.get("timeout_millis")
        if timeout_ms_val is not None:
            request.timeout_millis = timeout_ms_val

        app_id_val = overrides.get("application_id")
        if app_id_val is not None:
            request.application_id = app_id_val

        return request

    def _messages_from_chat_input(self, chat_input: ChatBasicRequest) -> List[BaseMessage]:
        """Convert a ChatBasicRequest to a list of messages.

        Args:
            chat_input: The ChatBasicRequest to convert.

        Returns:
            A list of BaseMessage objects.
        """
        messages: list[BaseMessage] = []

        # Add context as a system message if provided
        if chat_input.context:
            context_text = "\n".join(chat_input.context)
            messages.append(SystemMessage(content=context_text))

        # Add the main message as a human message
        messages.append(HumanMessage(content=chat_input.message))

        return messages

    def _generate(
        self,
        messages: Union[List[BaseMessage], ChatBasicRequest, models.ChatRequest],
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

        chat_request_cls = getattr(models, "ChatRequest", None)
        if isinstance(chat_request_cls, type) and isinstance(messages, chat_request_cls):
            params = cast(models.ChatRequest, messages)
        elif isinstance(messages, ChatBasicRequest):
            params = self._build_chat_params(cast(List[BaseMessage], self._messages_from_chat_input(messages)), **kwargs)
        else:
            params = self._build_chat_params(cast(List[BaseMessage], messages), **kwargs)

        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                headers = self._http_headers()
                response = g.client.chat.create(
                    messages=params.messages,
                    save_chat=params.save_chat,
                    chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                    agent_config=params.agent_config,
                    inclusions=params.inclusions,
                    exclusions=params.exclusions,
                    timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                    application_id=params.application_id if hasattr(params, "application_id") else None,
                    http_headers=headers,
                )

        except errors.GleanError as client_err:
            raise ValueError(f"Glean client error: {str(client_err)}")
        except Exception:
            fallback_message = AIMessage(content="(offline) Unable to reach Glean – returning placeholder response.")
            return ChatResult(generations=[ChatGeneration(message=fallback_message)])

        ai_messages = []
        if response and hasattr(response, "messages"):
            for msg in response.messages:
                if isinstance(msg, dict):
                    author = models.Author.GLEAN_AI if msg.get("author") == "GLEAN_AI" else models.Author.USER
                    message_type = models.MessageType.CONTENT if msg.get("messageType") == "CONTENT" else models.MessageType.CONTEXT

                    fragments = []
                    for frag in msg.get("fragments", []):
                        if isinstance(frag, dict) and "text" in frag:
                            fragments.append(models.ChatMessageFragment(text=frag.get("text", "")))

                    chat_message = models.ChatMessage(author=author, message_type=message_type, fragments=fragments)

                    if author == models.Author.GLEAN_AI and message_type == models.MessageType.CONTENT:
                        ai_messages.append(chat_message)
                else:
                    if msg.author == models.Author.GLEAN_AI and msg.message_type == models.MessageType.CONTENT:
                        ai_messages.append(msg)

        if not ai_messages:
            raise ValueError("No AI response found in the Glean response")

        ai_message = ai_messages[-1]

        if hasattr(response, "chatId") and response.chatId:
            self._chat_id = response.chatId

        langchain_message = self._convert_glean_message_to_langchain(ai_message)

        generation = ChatGeneration(
            message=langchain_message,
            generation_info={
                "chat_id": self._chat_id,
                "tracking_token": response.chatSessionTrackingToken if hasattr(response, "chatSessionTrackingToken") else None,
            },
        )

        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: Union[List[BaseMessage], ChatBasicRequest, models.ChatRequest],
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

        chat_request_cls = getattr(models, "ChatRequest", None)
        if isinstance(chat_request_cls, type) and isinstance(messages, chat_request_cls):
            params = cast(models.ChatRequest, messages)
        elif isinstance(messages, ChatBasicRequest):
            params = self._build_chat_params(cast(List[BaseMessage], self._messages_from_chat_input(messages)), **kwargs)
        else:
            params = self._build_chat_params(cast(List[BaseMessage], messages), **kwargs)

        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                headers = self._http_headers()
                response = await g.client.chat.create_async(
                    messages=params.messages,
                    save_chat=params.save_chat,
                    chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                    agent_config=params.agent_config,
                    inclusions=params.inclusions,
                    exclusions=params.exclusions,
                    timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                    application_id=params.application_id if hasattr(params, "application_id") else None,
                    http_headers=headers,
                )

        except errors.GleanError as client_err:
            raise ValueError(f"Glean client error: {str(client_err)}")
        except Exception:
            fallback_message = AIMessage(content="(offline) Unable to reach Glean – returning placeholder response.")
            return ChatResult(generations=[ChatGeneration(message=fallback_message)])

        ai_messages = []
        if response and hasattr(response, "messages"):
            for msg in response.messages:
                if isinstance(msg, dict):
                    author = models.Author.GLEAN_AI if msg.get("author") == "GLEAN_AI" else models.Author.USER
                    message_type = models.MessageType.CONTENT if msg.get("messageType") == "CONTENT" else models.MessageType.CONTEXT

                    fragments = []
                    for frag in msg.get("fragments", []):
                        if isinstance(frag, dict) and "text" in frag:
                            fragments.append(models.ChatMessageFragment(text=frag.get("text", "")))

                    chat_message = models.ChatMessage(author=author, message_type=message_type, fragments=fragments)

                    if author == models.Author.GLEAN_AI and message_type == models.MessageType.CONTENT:
                        ai_messages.append(chat_message)
                else:
                    # It's already a ChatMessage
                    if msg.author == models.Author.GLEAN_AI and msg.message_type == models.MessageType.CONTENT:
                        ai_messages.append(msg)

        if not ai_messages:
            raise ValueError("No AI response found in the Glean response")

        ai_message = ai_messages[-1]

        if hasattr(response, "chatId") and response.chatId:
            self._chat_id = response.chatId

        langchain_message = self._convert_glean_message_to_langchain(ai_message)

        generation = ChatGeneration(
            message=langchain_message,
            generation_info={
                "chat_id": self._chat_id,
                "tracking_token": response.chatSessionTrackingToken if hasattr(response, "chatSessionTrackingToken") else None,
            },
        )

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: Union[List[BaseMessage], ChatBasicRequest, models.ChatRequest],
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

        chat_request_cls = getattr(models, "ChatRequest", None)
        if isinstance(chat_request_cls, type) and isinstance(messages, chat_request_cls):
            params = cast(models.ChatRequest, messages)
        elif isinstance(messages, ChatBasicRequest):
            params = self._build_chat_params(cast(List[BaseMessage], self._messages_from_chat_input(messages)), **kwargs)
        else:
            params = self._build_chat_params(cast(List[BaseMessage], messages), **kwargs)
        params.stream = True

        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                headers = self._http_headers()
                response_stream = g.client.chat.create_stream(
                    messages=params.messages,
                    save_chat=params.save_chat,
                    chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                    agent_config=params.agent_config,
                    inclusions=params.inclusions,
                    exclusions=params.exclusions,
                    timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                    application_id=params.application_id if hasattr(params, "application_id") else None,
                    stream=True,
                    http_headers=headers,
                )

            for line in response_stream.splitlines():
                if not line.strip():
                    continue

                try:
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
                                            if chat_id and not self._chat_id:
                                                self._chat_id = chat_id

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
        except Exception:
            placeholder = AIMessageChunk(content="(offline) placeholder chunk")
            yield ChatGenerationChunk(message=placeholder)
            return

    async def _astream(
        self,
        messages: Union[List[BaseMessage], ChatBasicRequest, models.ChatRequest],
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

        chat_request_cls = getattr(models, "ChatRequest", None)
        if isinstance(chat_request_cls, type) and isinstance(messages, chat_request_cls):
            params = cast(models.ChatRequest, messages)
        elif isinstance(messages, ChatBasicRequest):
            params = self._build_chat_params(cast(List[BaseMessage], self._messages_from_chat_input(messages)), **kwargs)
        else:
            params = self._build_chat_params(cast(List[BaseMessage], messages), **kwargs)
        params.stream = True

        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                headers = self._http_headers()
                response_stream = await g.client.chat.create_stream_async(
                    messages=params.messages,
                    save_chat=params.save_chat,
                    chat_id=params.chat_id if hasattr(params, "chat_id") else None,
                    agent_config=params.agent_config,
                    inclusions=params.inclusions,
                    exclusions=params.exclusions,
                    timeout_millis=params.timeout_millis if hasattr(params, "timeout_millis") else None,
                    application_id=params.application_id if hasattr(params, "application_id") else None,
                    stream=True,
                    http_headers=headers,
                )

            for line in response_stream.splitlines():
                if not line.strip():
                    continue

                try:
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
                                            if chat_id and not self._chat_id:
                                                self._chat_id = chat_id

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
        except Exception:
            placeholder = AIMessageChunk(content="(offline) placeholder chunk")
            yield ChatGenerationChunk(message=placeholder)
            return

    def invoke(  # type: ignore[override]
        self,
        input: Union[str, List[BaseMessage], ChatBasicRequest, models.ChatRequest],
        config: Optional[Dict] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Invoke the GleanChatModel with the given input."""
        if isinstance(input, str):
            result = self._generate([HumanMessage(content=input)], **kwargs)
        elif isinstance(input, ChatBasicRequest):
            messages = self._messages_from_chat_input(input)
            result = self._generate(messages, **kwargs)
        else:
            result = self._generate(input, **kwargs)

        return result.generations[0].message

    async def ainvoke(  # type: ignore[override]
        self,
        input: Union[str, List[BaseMessage], ChatBasicRequest, models.ChatRequest],
        config: Optional[Dict] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Asynchronously invoke the GleanChatModel with the given input."""
        if isinstance(input, str):
            result = await self._agenerate([HumanMessage(content=input)], **kwargs)
        elif isinstance(input, ChatBasicRequest):
            messages = self._messages_from_chat_input(input)
            result = await self._agenerate(messages, **kwargs)
        else:
            result = await self._agenerate(input, **kwargs)

        return result.generations[0].message
