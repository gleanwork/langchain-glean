---
name: langchain-glean
description: Use the langchain-glean package to call Glean from LangChain — chat models (ChatGlean / ChatGleanAgent), retrievers over Glean search and the people directory, and Glean tools for LangChain agents. Load when writing Python that imports langchain_glean or wires Glean into a LangChain app, chain, or agent.
---

# langchain-glean

LangChain integration for [Glean](https://www.glean.com/). It wraps the Glean Python SDK (`glean-api-client`) behind LangChain's `Runnable` interface, exposing Glean as chat models, retrievers, and agent tools.

## When to use

Load this skill when writing Python that imports `langchain_glean`, or when wiring Glean into a LangChain app — a Glean-backed chat model, a retriever over Glean search or the people directory, or Glean tools inside a tool-calling agent. For talking to the Glean API *without* LangChain, use `glean-api-client` directly instead.

## Install & import

```bash
pip install -U langchain-glean
```

Every public entry point is re-exported from the top-level package:

```python
from langchain_glean import (
    ChatGlean, ChatGleanAgent,
    GleanSearchRetriever, GleanPeopleProfileRetriever,
    GleanToolkit,
    GleanSearchTool, GleanPeopleProfileSearchTool, GleanChatTool,
    GleanListAgentsTool, GleanGetAgentSchemaTool, GleanRunAgentTool,
)
```

The simple request models live beside their component: `from langchain_glean.chat_models import ChatBasicRequest`, `from langchain_glean.retrievers import SearchBasicRequest, PeopleProfileBasicRequest`.

## Authoritative API

The package ships a `py.typed` marker and is fully typed (mypy strict). Read the inline type hints in the installed package for exact constructor arguments, request-model fields, and return types — do not guess them:

- the component classes under `langchain_glean/chat_models/`, `retrievers/`, `tools/`, and `toolkit.py`
- the simple request models (`ChatBasicRequest`, `SearchBasicRequest`, `PeopleProfileBasicRequest`) defined alongside them
- for full-control requests, the Glean SDK models in the `glean-api-client` dependency (`from glean.api_client import models`) — read that package's types too

Components implement LangChain's `Runnable` interface, so `invoke` / `ainvoke` / `stream` / `astream` come from `langchain-core`; defer to its docs for the runnable surface.

## Usage patterns

- **Auth is environment-first.** Set `GLEAN_API_TOKEN` and `GLEAN_SERVER_URL` (the full backend URL; `GLEAN_INSTANCE` is a deprecated fallback). With a *global* token, also set `GLEAN_ACT_AS` to the user to act as. All components share this resolution through a common mixin, so you don't build an SDK client by hand — set the env (or pass the same values as constructor args) and construct the component.
- **Three input styles; escalate only when needed.** (1) a plain query string (retrievers), (2) a `*BasicRequest` Pydantic model for the common options, (3) a full `glean.api_client` model (`SearchRequest`, `ChatRequest`, `ListEntitiesRequest`) for complete control. Start at (1)/(2); reach for (3) only for fields the basic model doesn't expose.
- **Chat.** `ChatGlean` wraps the Glean Assistant; `ChatGleanAgent(agent_id=...)` targets one specific agent. Both take LangChain messages and return LangChain messages; use `stream`/`astream` to stream, and resume a conversation with `chat_id`.
- **Retrievers** return LangChain `Document`s from Glean search or the people directory — drop them straight into a RAG chain.
- **Agent tools** wrap the same capabilities for tool-calling agents, and `GleanToolkit` aggregates them. The agent-discovery tools are meant to be chained in order: `GleanListAgentsTool` → `GleanGetAgentSchemaTool` → `GleanRunAgentTool`.
- **Async is everywhere** — every retriever, tool, and chat model exposes `ainvoke`/`astream`.

## Common mistakes

- **Hand-constructing a Glean client or threading tokens through each call.** Auth resolves from env/constructor via the shared mixin; configure once, then just construct the component.
- **Using `GLEAN_INSTANCE` in new code** — it's a deprecated fallback; prefer the full `GLEAN_SERVER_URL`.
- **Omitting `GLEAN_ACT_AS` with a global token** — global tokens must name the acting user or calls fail. (User-scoped tokens don't need it.)
- **Guessing request-model field names.** `*BasicRequest` and the `glean.api_client` models are typed — read the hints instead of inventing fields.
- **Reaching for a full SDK request object when a string or `*BasicRequest` would do** — extra complexity for no gain.
- **Confusing the chat classes:** `ChatGlean` is the Assistant; `ChatGleanAgent` is one agent selected by `agent_id`.

## Version notes

Check the installed version with `pip show langchain-glean` or `python -c "import langchain_glean; print(langchain_glean.__version__)"`. It pins `glean-api-client` (the generated Glean Python SDK) and `langchain-core`; after upgrading either, re-read the types for changed request-model fields rather than trusting older snippets, and consult `CHANGELOG.md` for breaking changes. Requires Python 3.9+.
