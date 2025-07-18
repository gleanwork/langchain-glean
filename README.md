# langchain-glean

[![PyPI version](https://badge.fury.io/py/glean-api-client.svg)](https://badge.fury.io/py/glean-api-client)

Connect [Glean](https://www.glean.com/) – The Work AI platform connected to all your data – with [LangChain](https://github.com/langchain-ai/langchain).

## Overview

The package provides:

* **Chat models** – `ChatGlean` wraps the `/v1/chat` assistant API and `ChatGleanAgent` targets specific agents
* **Retrievers** – typed helpers for Glean search and the people directory
* **Tools** – drop-in utilities for LangChain agents

Each implementation supports **three input styles** so you can start simple and scale up only when required:

1. **Plain strings (retrievers only)** – pass the search query text
2. **Simple objects** – pass a small Pydantic model (e.g. `ChatBasicRequest`, `SearchBasicRequest`, `PeopleProfileBasicRequest`) 
3. **Full Glean request classes** – hand-craft a `glean.models.SearchRequest`, `glean.models.ChatRequest`, or `glean.models.ListEntitiesRequest` 

## Installation

```bash
pip install -U langchain-glean
```

### Environment variables

```bash
export GLEAN_API_TOKEN="<your-token>"   # user or global token
export GLEAN_INSTANCE="acme"            # Glean instance name
export GLEAN_ACT_AS="user@acme.com"     # only for global tokens
```

## Quick Start

### Chat with Glean Assistant

```python
from langchain_core.messages import HumanMessage
from langchain_glean.chat_models import ChatGlean

chat = ChatGlean()
response = chat.invoke([HumanMessage(content="When is the next company holiday?")])
print(response.content)
```

Need streaming? Replace `invoke` with `stream` or `astream`.

## Chat Models

### ChatGlean

Connect to the Glean Assistant API with LangChain's chat interface.

#### Basic usage with message list

```python
from langchain_glean.chat_models import ChatGlean
from langchain_core.messages import HumanMessage

chat = ChatGlean()
response = chat.invoke([HumanMessage(content="Hello")])
```

#### Using the simplified request object

```python
from langchain_glean.chat_models import ChatGlean, ChatBasicRequest

chat = ChatGlean()
response = chat.invoke(ChatBasicRequest(
    message="Hello", 
    context=["Previous conversation context"]
))
```

#### Using the full Glean SDK request

```python
from glean.api_client import models
from langchain_glean.chat_models import ChatGlean

chat = ChatGlean()
req = models.ChatRequest(
    messages=[
        models.ChatMessage(
            author="USER", 
            message_type="CONTENT", 
            fragments=[models.ChatMessageFragment(text="Hello")]
        )
    ]
)
response = chat.invoke(req)
```

### ChatGleanAgent

Connect directly to an individual Glean agent.

```python
from langchain_core.messages import HumanMessage
from langchain_glean.chat_models import ChatGleanAgent

agent_chat = ChatGleanAgent(agent_id="abc123")
response = agent_chat.invoke([HumanMessage(content="What are our Q4 sales targets?")])
print(response.content)
```

You can provide additional input fields for the agent:

```python
response = agent_chat.invoke(
    [HumanMessage(content="What are our sales targets?")],
    fields={"department": "Marketing", "quarter": "Q4"}
)
```

## Retrievers

### GleanSearchRetriever

Search Glean's unified index and get results as LangChain documents.

#### Simple string query

```python
from langchain_glean.retrievers import GleanSearchRetriever

retriever = GleanSearchRetriever()
results = retriever.invoke("quarterly report")
```

#### Using the simplified request object

```python
from langchain_glean.retrievers import GleanSearchRetriever, SearchBasicRequest

retriever = GleanSearchRetriever()
results = retriever.invoke(SearchBasicRequest(
    query="quarterly report", 
    data_sources=["confluence", "drive"]
))
```

#### Using the full Glean SDK request

```python
from glean.api_client import models
from langchain_glean.retrievers import GleanSearchRetriever

retriever = GleanSearchRetriever()
req = models.SearchRequest(
    query="quarterly report", 
    page_size=5,
    facet_filters=[models.FacetFilter(name="datasource", values=["confluence"])]
)
results = retriever.invoke(req)
```

### GleanPeopleProfileRetriever

Search Glean's people directory and get results as LangChain documents.

#### Simple string query

```python
from langchain_glean.retrievers import GleanPeopleProfileRetriever

people = GleanPeopleProfileRetriever()
results = people.invoke("jane doe")
```

#### Using the simplified request object

```python
from langchain_glean.retrievers import GleanPeopleProfileRetriever, PeopleProfileBasicRequest

people = GleanPeopleProfileRetriever()
results = people.invoke(PeopleProfileBasicRequest(
    query="staff engineer", 
    page_size=3
))
```

#### Using the full Glean SDK request

```python
from glean.api_client import models
from langchain_glean.retrievers import GleanPeopleProfileRetriever

people = GleanPeopleProfileRetriever()
req = models.ListEntitiesRequest(
    entity_type="PEOPLE", 
    query="staff engineer", 
    page_size=3
)
results = people.invoke(req)
```

## Tools for LangChain Agents

### Available Tools

| Tool name | Purpose | Arguments |
|-----------|---------|-----------|
| `GleanSearchTool` | Search content | Query string or SearchBasicRequest |
| `GleanPeopleProfileSearchTool` | Find people | Query string or PeopleProfileBasicRequest |
| `GleanChatTool` | Converse with Glean Assistant | Message string or ChatBasicRequest |
| `GleanListAgentsTool` | List available agents | None |
| `GleanGetAgentSchemaTool` | Get agent input schema | `agent_id` (str) |
| `GleanRunAgentTool` | Run a specific agent | `agent_id` (str), `fields` (dict) |

### Basic Search Tool Example

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool

# Create the tool
retriever = GleanSearchRetriever()
search_tool = GleanSearchTool(retriever=retriever)

# Set up the agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You can search our knowledge base when needed."),
    ("user", "{input}"),
])

agent = create_openai_tools_agent(llm, [search_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[search_tool])

# Run the agent
result = executor.invoke({"input": "Find the latest QBR deck"})
print(result["output"])
```

### Using Glean Agent Tools

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_glean.tools import GleanListAgentsTool, GleanGetAgentSchemaTool, GleanRunAgentTool

# Create tools
list_agents_tool = GleanListAgentsTool()
get_schema_tool = GleanGetAgentSchemaTool()
run_agent_tool = GleanRunAgentTool()

# Set up the agent
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You can find and run Glean agents to answer questions."),
    ("user", "{input}"),
])

agent = create_openai_tools_agent(llm, [list_agents_tool, get_schema_tool, run_agent_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[list_agents_tool, get_schema_tool, run_agent_tool])

# Run the agent
result = executor.invoke({
    "input": "Find an agent that can help with sales data and run it to get Q4 forecast"
})
print(result["output"])
```

## Advanced Usage

### Full Request Objects

Pass any SDK request class (`SearchRequest`, `ChatRequest`, `ListEntitiesRequest`) directly for complete API control.

### Async Support

Every retriever and tool exposes `ainvoke` and async streams:

```python
# Async search
documents = await retriever.ainvoke("monthly revenue")

# Async streaming chat
async for chunk in chat.astream([HumanMessage(content="Hello")]):
    print(chunk.content, end="", flush=True)
```

### Custom Agent Config

Override model behavior per call:

```python
chat.invoke(
    ChatBasicRequest(message="Summarise last quarter"),
    agent_config={"agent": "GPT", "mode": "SEARCH"},
    timeout_millis=30_000,
)
```

### Resume a Chat

Either pass `chat_id` per call or set it as a property:

```python
chat = ChatGlean()
chat.chat_id = "abc123"
chat.invoke([HumanMessage(content="Continue...")])
```

## Contributing

1. `mise install && task setup`
2. `task lint && task test`
3. Open a PR!

## Links

* Glean API – <https://developer.glean.com>
* LangChain docs – <https://python.langchain.com>
* Source code – <https://github.com/langchain-ai/langchain-glean>
