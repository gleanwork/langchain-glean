# langchain-glean

Connect [Glean](https://www.glean.com/) – The Work AI platform connected to all your data – with [LangChain](https://github.com/langchain-ai/langchain).

The package provides:

* **Chat model** – `ChatGlean` wraps the `/v1/chat` assistant API.
* **Retrievers** – typed helpers for Glean search and the people directory.
* **Tools** – drop-in utilities for LangChain agents.

Each implementation supports **three input styles** so you can start simple and scale up only when required:

1. **Plain strings (retrievers only)** – pass the search query text.
2. **Simple objects** – pass a small Pydantic model (e.g. `ChatBasicRequest`, `SearchBasicRequest`, `PeopleProfileBasicRequest`).  These cover the most common parameters.
3. **Full Glean request classes** – hand-craft a `glean.models.SearchRequest`, `glean.modelsChatRequest`, or `glean.modelsListEntitiesRequest` when you need every available field.

The library auto-detects which form you provided and forwards it unchanged to the Glean SDK, giving you progressive control with zero boilerplate.

---

## 📦 Installation

```bash
pip install -U langchain-glean
```

### Environment variables

```bash
export GLEAN_API_TOKEN="<your-token>"   # user or global token
export GLEAN_INSTANCE="acme"            # Glean instance name
export GLEAN_ACT_AS="user@acme.com"     # only for global tokens
```

---

## Quick start – Chat

```python
from langchain_core.messages import HumanMessage
from langchain_glean.chat_models import ChatGlean

chat = ChatGlean()
response = chat.invoke([HumanMessage(content="When is the next company holiday?")])
print(response.content)
```

Need streaming? Replace `invoke` with `stream` or `astream`.

---

## Components

### 1. Chat model – `ChatGlean`

| Feature                | Notes |
|------------------------|-------|
| Basic request          | `ChatBasicRequest(message, context=None)` |
| Escape hatch           | Pass a fully-populated `glean.models.ChatRequest` |
| Per-call overrides     | `save_chat`, `agent_config`, `timeout_millis`, `application_id`, `inclusions` / `exclusions` |

| Input style | Example |
|-------------|---------|
| String list | ```python
from langchain_glean.chat_models import ChatGlean
from langchain_core.messages import HumanMessage

chat = ChatGlean()
response = chat.invoke([HumanMessage(content="Hello")])
``` |
| Simple object | ```python
from langchain_glean.chat_models import ChatGlean, ChatBasicRequest

chat = ChatGlean()
response = chat.invoke(ChatBasicRequest(message="Hello", context=["Hi there"]))
``` |
| Full request | ```python
from glean import models
from langchain_glean.chat_models import ChatGlean

chat = ChatGlean()
req = models.ChatRequest(messages=[models.ChatMessage(author="USER", message_type="CONTENT", fragments=[models.ChatMessageFragment(text="Hello")])])
response = chat.invoke(req)
``` |

### 2. Search retriever – `GleanSearchRetriever`

| Path              | Schema |
|-------------------|--------|
| Recommended       | `SearchBasicRequest(query, data_sources=None)` |
| Escape hatch      | `glean.models.SearchRequest` |

| Input style | Example |
|-------------|---------|
| String query | ```python
retriever = GleanSearchRetriever()
results = retriever.invoke("quarterly report")
``` |
| Simple object | ```python
from langchain_glean.retrievers import GleanSearchRetriever, SearchBasicRequest

retriever = GleanSearchRetriever()
results = retriever.invoke(SearchBasicRequest(query="quarterly report", data_sources=["confluence"]))
``` |
| Full request | ```python
from glean import models
from langchain_glean.retrievers import GleanSearchRetriever

retriever = GleanSearchRetriever()
req = models.SearchRequest(query="quarterly report", page_size=5)
results = retriever.invoke(req)
``` |

### 3. People directory retriever – `PeopleProfileRetriever`

```python
from langchain_glean.retrievers import PeopleProfileRetriever, PeopleProfileBasicRequest

people = PeopleProfileRetriever()
results = people.invoke(PeopleProfileBasicRequest(query="staff engineer", page_size=5))
print(results[0].page_content)
```

| Input style | Example |
|-------------|---------|
| String query | ```python
people = PeopleProfileRetriever()
results = people.invoke("jane doe")
``` |
| Simple object | ```python
from langchain_glean.retrievers import PeopleProfileRetriever, PeopleProfileBasicRequest

people = PeopleProfileRetriever()
results = people.invoke(PeopleProfileBasicRequest(query="staff engineer", page_size=3))
``` |
| Full request | ```python
from glean import models
from langchain_glean.retrievers import PeopleProfileRetriever

people = PeopleProfileRetriever()
req = models.ListEntitiesRequest(entity_type="PEOPLE", query="staff engineer", page_size=3)
results = people.invoke(req)
``` |

---

## Tools for agents

| Tool name               | Purpose                           | Args schema |
|-------------------------|-----------------------------------|-------------|
| `glean_search`          | Search content                    | `SearchBasicRequest` |
| `people_profile_search` | Find people                       | `PeopleProfileBasicRequest` |
| `chat`                  | Converse with Glean Assistant     | `ChatBasicRequest` |

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool

retriever = GleanSearchRetriever()
search_tool = GleanSearchTool(retriever=retriever)

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You can search our knowledge base when needed."),
    ("user", "{input}"),
])

agent = create_openai_tools_agent(llm, [search_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[search_tool])
print(executor.invoke({"input": "Find the latest QBR deck"})["output"])
```

---

## Advanced usage

* **Full request objects** – pass any SDK request class (`SearchRequest`, `ChatRequest`, `ListEntitiesRequest`) directly for 100 % API coverage.
* **Async everywhere** – every retriever/tool exposes `ainvoke` and async streams.
* **Custom agent config** – override model behaviour per call:

```python
chat.invoke(
    ChatBasicRequest(message="Summarise last quarter"),
    agent_config={"agent": "GPT", "mode": "SEARCH"},
    timeout_millis=30_000,
)
```

* **Resume a chat** – either pass ``chat_id=...`` per call **or** set the property::

```python
chat = ChatGlean()
chat.chat_id = "abc123"
chat.invoke([HumanMessage(content="Continue...")])
```

---

## Contributing

1. `uv sync`
2. `pytest && ruff check . && mypy .`
3. Open a PR!

---

## Links

* Glean API – <https://developer.glean.com>
* LangChain docs – <https://python.langchain.com>
* Source code – <https://github.com/langchain-ai/langchain-glean>
