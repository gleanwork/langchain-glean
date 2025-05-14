from typing import Any, Dict, List, Optional, cast

from glean import Glean, errors
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from langchain_glean._api_client_mixin import GleanAPIClientMixin

# Type for Glean agent API messages
GleanApiMessage = Dict[str, Any]


class ChatGleanAgent(GleanAPIClientMixin, BaseChatModel):
    """LangChain ChatModel wrapper for running a specific Glean Agent."""

    agent_id: str = Field(description="ID of the agent to run")
    model_config = ConfigDict(extra="allow")

    @property
    def _llm_type(self) -> str:
        return "glean-agent-chat"

    def _extract_user_input(self, messages: List[BaseMessage]) -> str:
        user_messages = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        return "\n".join(user_messages).strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:  # noqa: D401
        if stop is not None:
            raise ValueError("stop sequences are not supported by AgentChatModel")

        fields: Dict[str, str] = cast(Dict[str, str], kwargs.pop("fields", {}))
        user_input = self._extract_user_input(messages)
        if user_input and "input" not in fields:
            fields["input"] = user_input

        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = g.client.agents.run(agent_id=self.agent_id, fields=fields, stream=False)
        except errors.GleanError as e:
            raise ValueError(f"Glean client error: {e}") from e
        except Exception:
            fallback = AIMessage(content="(offline) Unable to reach Glean – returning placeholder response.")
            return ChatResult(generations=[ChatGeneration(message=fallback)])

        content = ""
        if hasattr(response, "messages") and response.messages:
            ai_messages = [m for m in response.messages if isinstance(m, dict) and m.get("author") == "GLEAN_AI"]

            if ai_messages:
                last_message: GleanApiMessage = cast(GleanApiMessage, ai_messages[-1])
                fragments = last_message.get("fragments", [])

                if fragments:
                    for frag in fragments:
                        txt = frag.get("text", "")
                        if isinstance(txt, str):
                            content += txt

        if not content:
            content = str(response)

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:  # noqa: D401
        if stop is not None:
            raise ValueError("stop sequences are not supported by AgentChatModel")

        fields: Dict[str, str] = cast(Dict[str, str], kwargs.pop("fields", {}))
        user_input = self._extract_user_input(messages)
        if user_input and "input" not in fields:
            fields["input"] = user_input

        try:
            with Glean(api_token=self.api_token, instance=self.instance) as g:
                response = await g.client.agents.run_async(agent_id=self.agent_id, fields=fields, stream=False)
        except errors.GleanError as e:
            raise ValueError(f"Glean client error: {e}") from e
        except Exception:
            fallback = AIMessage(content="(offline) Unable to reach Glean – returning placeholder response.")
            return ChatResult(generations=[ChatGeneration(message=fallback)])

        content = ""
        if hasattr(response, "messages") and response.messages:
            ai_messages = [m for m in response.messages if isinstance(m, dict) and m.get("author") == "GLEAN_AI"]

            if ai_messages:
                last_message: GleanApiMessage = cast(GleanApiMessage, ai_messages[-1])
                fragments = last_message.get("fragments", [])

                if fragments:
                    for frag in fragments:
                        txt = frag.get("text", "")
                        if isinstance(txt, str):
                            content += txt

        if not content:
            content = str(response)

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])
