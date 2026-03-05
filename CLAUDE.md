# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`langchain-glean` is a LangChain integration package for [Glean](https://www.glean.com/), the enterprise Work AI platform. It provides chat models, retrievers, and tools that connect Glean's APIs with LangChain's abstractions.

## Development Commands

```bash
# Setup (install mise first: curl https://mise.run | sh)
mise install && mise run setup

# Testing
mise run test              # Run unit tests (network disabled)
mise run test:watch        # Run tests in watch mode
mise run test:integration  # Run integration tests (requires credentials)
mise run test:all          # Run all tests + lint fixes + mypy

# Linting
mise run lint              # Run all linters (ruff + format check + mypy)
mise run lint:fix          # Auto-fix lint issues and format code

# Release (maintainers only)
DRY_RUN=true mise run release  # Preview version bump
mise run release               # Bump version, update CHANGELOG.md, create tag
```

Run a single test with: `uv run pytest tests/unit_tests/test_glean_search_retriever.py::TestGleanSearchRetriever::test_init -v`

## Architecture

### Three-Input-Style Pattern

All major components (chat models, retrievers, tools) support three input styles for flexibility:

1. **Plain strings** (retrievers only) - simple query text
2. **Basic request objects** - small Pydantic models (`ChatBasicRequest`, `SearchBasicRequest`, `PeopleProfileBasicRequest`)
3. **Full SDK requests** - `glean.models.ChatRequest`, `glean.models.SearchRequest`, `glean.models.ListEntitiesRequest`

### Core Components

```
langchain_glean/
├── _api_client_mixin.py   # GleanAPIClientMixin: shared auth (instance, api_token, act_as)
├── chat_models/
│   ├── chat.py            # ChatGlean: wraps /v1/chat assistant API
│   └── agent_chat.py      # ChatGleanAgent: targets specific Glean agents
├── retrievers/
│   ├── search.py          # GleanSearchRetriever: unified search index
│   └── people.py          # GleanPeopleProfileRetriever: people directory
├── tools/                 # LangChain tool wrappers for agent use
│   ├── search.py          # GleanSearchTool
│   ├── chat.py            # GleanChatTool
│   ├── people_profile_search.py
│   ├── list_agents.py     # GleanListAgentsTool
│   ├── get_agent_schema.py
│   └── run_agent.py       # GleanRunAgentTool
└── toolkit.py             # GleanToolkit: aggregates all tools
```

### Key Patterns

- **GleanAPIClientMixin**: All components inherit from this mixin, which resolves `GLEAN_SERVER_URL` (preferred) or `GLEAN_INSTANCE`, `GLEAN_API_TOKEN`, and `GLEAN_ACT_AS` from environment variables or constructor args. Use `_build_glean_client()` to create SDK clients.
- **Async support**: Every retriever/tool exposes `ainvoke`, `astream` via the standard LangChain async interface.
- **Message conversion**: `ChatGlean._convert_message_to_glean_format()` maps LangChain messages to Glean's `ChatMessage` format (author, message_type, fragments).

## Testing

- Unit tests mock the Glean SDK at the module boundary (e.g., `patch("langchain_glean.retrievers.search.Glean")`)
- Network is disabled via `--disable-socket` for unit tests
- Integration tests require real `GLEAN_API_TOKEN` and `GLEAN_SERVER_URL` (or `GLEAN_INSTANCE`) environment variables

## Dependencies

See `pyproject.toml` for current dependency versions. Key dependencies:
- `glean-api-client` - Glean Python SDK
- `langchain-core` - LangChain core abstractions
- Python 3.9+
