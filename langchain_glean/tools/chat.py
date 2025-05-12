from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import BaseTool

from langchain_glean.chat_models.chat import ChatBasicRequest, ChatGlean


class GleanChatTool(BaseTool):
    """Tool that sends a chat message to Glean Assistant and returns the response text."""

    name: str = "chat"
    description: str = "Interact with Glean's AI assistant using a message and optional context."

    args_schema: type = ChatBasicRequest

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        try:
            if args and not kwargs:
                tool_input = args[0]
            elif "message" in kwargs or "context" in kwargs:
                tool_input = {k: v for k, v in kwargs.items()}
            elif "input" in kwargs:
                tool_input = kwargs["input"]
            else:
                return "Error: No valid input provided"

            cg = ChatGlean()

            if isinstance(tool_input, str):
                return cg.invoke(tool_input).content  # type: ignore[return-value]

            if isinstance(tool_input, dict):
                message = tool_input.get("message", "")
                context = tool_input.get("context")
                other_kwargs = {k: v for k, v in tool_input.items() if k not in {"message", "context"}}
                if context:
                    return cg.invoke(ChatBasicRequest(message=message, context=context), **other_kwargs).content  # type: ignore[return-value]
                return cg.invoke(message, **other_kwargs).content  # type: ignore[return-value]

            if isinstance(tool_input, ChatBasicRequest):
                return cg.invoke(tool_input).content  # type: ignore[return-value]

            return f"Error: Unexpected input type {type(tool_input)}"
        except Exception as e:  # noqa: BLE001
            return f"Error running Glean chat: {str(e)}"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        try:
            if args and not kwargs:
                tool_input = args[0]
            elif "message" in kwargs or "context" in kwargs:
                tool_input = {k: v for k, v in kwargs.items()}
            elif "input" in kwargs:
                tool_input = kwargs["input"]
            else:
                return "Error: No valid input provided"

            cg = ChatGlean()

            if isinstance(tool_input, str):
                return (await cg.ainvoke(tool_input)).content  # type: ignore[return-value]

            if isinstance(tool_input, dict):
                message = tool_input.get("message", "")
                context = tool_input.get("context")
                other_kwargs = {k: v for k, v in tool_input.items() if k not in {"message", "context"}}
                if context:
                    return (await cg.ainvoke(ChatBasicRequest(message=message, context=context), **other_kwargs)).content  # type: ignore[return-value]
                return (await cg.ainvoke(message, **other_kwargs)).content  # type: ignore[return-value]

            if isinstance(tool_input, ChatBasicRequest):
                return (await cg.ainvoke(tool_input)).content  # type: ignore[return-value]

            return f"Error: Unexpected input type {type(tool_input)}"
        except Exception as e:  # noqa: BLE001
            return f"Error running Glean chat: {str(e)}"

    def invoke(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Override BaseTool.invoke to support legacy message/context kwargs."""
        if args or ("input" in kwargs):
            return super().invoke(*args, **kwargs)  # type: ignore[arg-type]

        # Convert message/context kwargs into a single dict input
        if "message" in kwargs or "context" in kwargs:
            inp: Dict[str, Any] = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in {"message", "context"}}
            return super().invoke(inp, **kwargs)  # type: ignore[arg-type]

        raise TypeError("invoke() missing required argument 'input'")

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if args or ("input" in kwargs):
            return await super().ainvoke(*args, **kwargs)  # type: ignore[arg-type]

        if "message" in kwargs or "context" in kwargs:
            inp: Dict[str, Any] = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in {"message", "context"}}
            return await super().ainvoke(inp, **kwargs)  # type: ignore[arg-type]

        raise TypeError("ainvoke() missing required argument 'input'")
