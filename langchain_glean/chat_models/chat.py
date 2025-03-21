import json
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
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
from pydantic import Field, PrivateAttr, model_validator

from langchain_glean.client import GleanAuth, GleanClient
from langchain_glean.client.glean_client import GleanClientError, GleanConnectionError, GleanHTTPError


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

    subdomain: str = Field(description="Subdomain for Glean instance")
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

    _auth: GleanAuth = PrivateAttr()
    _client: GleanClient = PrivateAttr()

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
        values["subdomain"] = get_from_dict_or_env(values, "subdomain", "GLEAN_SUBDOMAIN")
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
            self._auth = GleanAuth(api_token=self.api_token, subdomain=self.subdomain, act_as=self.act_as)
            self._client = GleanClient(auth=self._auth, timeout=self.timeout)
        except Exception as e:
            raise ValueError(f"Failed to initialize Glean client: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "glean-chat"

    def _convert_message_to_glean_format(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert a LangChain message to Glean's message format.

        Args:
            message: The LangChain message to convert.

        Returns:
            The message in Glean's format.
        """
        if isinstance(message, HumanMessage):
            author = "USER"
        elif isinstance(message, AIMessage):
            author = "GLEAN_AI"
        elif isinstance(message, SystemMessage):
            # System messages are treated as context messages in Glean
            author = "USER"
            return {"author": author, "messageType": "CONTEXT", "fragments": [{"text": message.content}]}
        elif isinstance(message, ChatMessage):
            # Map custom roles to Glean's format
            if message.role.upper() == "USER":
                author = "USER"
            elif message.role.upper() == "ASSISTANT" or message.role.upper() == "AI":
                author = "GLEAN_AI"
            else:
                # Default to USER for unknown roles
                author = "USER"
        else:
            # Default to USER for unknown message types
            author = "USER"

        return {"author": author, "messageType": "CONTENT", "fragments": [{"text": message.content}]}

    def _convert_glean_message_to_langchain(self, message: Dict[str, Any]) -> BaseMessage:
        """Convert a Glean message to a LangChain message.

        Args:
            message: The Glean message to convert.

        Returns:
            The message in LangChain's format.
        """
        author = message.get("author", "")
        fragments = message.get("fragments", [])

        content = ""
        for fragment in fragments:
            if "text" in fragment:
                content += fragment["text"]

        if author == "GLEAN_AI":
            return AIMessage(content=content)
        else:
            return HumanMessage(content=content)

    def _build_chat_params(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Create a chat request for the Glean API.

        Args:
            messages: The messages to include in the request.

        Returns:
            The chat request in Glean's format.
        """
        glean_messages = [self._convert_message_to_glean_format(msg) for msg in messages]

        # Build the base request with required parameters
        request = {"messages": glean_messages, "saveChat": self.save_chat, "agentConfig": self.agent_config}

        # Add optional parameters if they are set
        if self.chat_id:
            request["chatId"] = self.chat_id

        # Add additional parameters from the OpenAPI specification
        if self.inclusions:
            request["inclusions"] = self.inclusions

        if self.exclusions:
            request["exclusions"] = self.exclusions

        if self.timeout_millis:
            request["timeoutMillis"] = self.timeout_millis

        if self.application_id:
            request["applicationId"] = self.application_id

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
            response = self._client.post("chat", data=json.dumps(params), headers={"Content-Type": "application/json"}, timeout=self.timeout)
            chat_response = self._client.parse_response(response)
        except GleanHTTPError as http_err:
            error_details = f"HTTP Error {http_err.status_code}"
            if http_err.response:
                error_details += f": {http_err.response}"
            raise ValueError(f"Glean API error: {error_details}")
        except GleanConnectionError as conn_err:
            raise ValueError(f"Glean connection error: {str(conn_err)}")
        except GleanClientError as client_err:
            raise ValueError(f"Glean client error: {str(client_err)}")

        response_messages = chat_response.get("messages", [])
        ai_messages = [msg for msg in response_messages if msg.get("author") == "GLEAN_AI" and msg.get("messageType") == "CONTENT"]

        if not ai_messages:
            raise ValueError("No AI response found in the Glean response")

        ai_message = ai_messages[-1]

        if "chatId" in chat_response:
            self.chat_id = chat_response["chatId"]

        langchain_message = self._convert_glean_message_to_langchain(ai_message)

        generation = ChatGeneration(
            message=langchain_message, generation_info={"chat_id": self.chat_id, "tracking_token": chat_response.get("chatSessionTrackingToken", "")}
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

        params["stream"] = True

        try:
            response = self._client.post("chat", data=json.dumps(params), headers={"Content-Type": "application/json"}, stream=True, timeout=self.timeout)

            content_buffer = ""
            chat_id = None
            tracking_token = None

            for line in response.iter_lines():
                if not line:
                    continue

                line_text = line.decode("utf-8")
                if line_text.startswith("data: "):
                    line_text = line_text[6:]

                try:
                    chunk_data = json.loads(line_text)

                    if "chatId" in chunk_data and not chat_id:
                        chat_id = chunk_data["chatId"]
                        self.chat_id = chat_id

                    if "chatSessionTrackingToken" in chunk_data and not tracking_token:
                        tracking_token = chunk_data["chatSessionTrackingToken"]

                    for message in chunk_data.get("messages", []):
                        if message.get("author") == "GLEAN_AI" and message.get("messageType") == "CONTENT":
                            for fragment in message.get("fragments", []):
                                if "text" in fragment:
                                    new_content = fragment.get("text", "")
                                    if new_content:
                                        content_buffer += new_content

                                        message_chunk = AIMessageChunk(content=new_content)

                                        chunk = ChatGenerationChunk(
                                            message=message_chunk, generation_info={"chat_id": chat_id, "tracking_token": tracking_token}
                                        )
                                        yield chunk

                                        if run_manager:
                                            run_manager.on_llm_new_token(new_content)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    if run_manager:
                        run_manager.on_llm_error(e)
                    continue

        except GleanHTTPError as http_err:
            error_details = f"HTTP Error {http_err.status_code}"
            if http_err.response:
                error_details += f": {http_err.response}"
            raise ValueError(f"Glean API error: {error_details}")
        except GleanConnectionError as conn_err:
            raise ValueError(f"Glean connection error: {str(conn_err)}")
        except GleanClientError as client_err:
            raise ValueError(f"Glean client error: {str(client_err)}")
        except Exception as e:
            raise ValueError(f"Error during streaming: {str(e)}")
