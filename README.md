# langchain-glean

This package contains the LangChain integration with [Glean](https://www.glean.com/), an enterprise search platform. It allows you to search and retrieve information from your organization's content using LangChain.

## Installation

```bash
pip install -U langchain-glean
```

## Configuration

You need to configure your Glean credentials by setting the following environment variables:

```bash
export GLEAN_SUBDOMAIN="your-glean-subdomain"
export GLEAN_API_TOKEN="your-api-token"
export GLEAN_ACT_AS="user@example.com"  # Optional: Email to act as when making requests
```

## Usage

### Using the Chat Model

The `ChatGlean` allows you to interact with Glean's AI chat functionality:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_glean import ChatGlean

# Initialize the chat model (will use environment variables)
chat = ChatGlean()

# Create messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What are the company holidays this year?")
]

# Generate a response
response = chat.invoke(messages)
print(response.content)
```

#### Streaming Responses

You can stream responses from the chat model:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_glean import ChatGlean

# Initialize the chat model
chat = ChatGlean()

# Create messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain retrieval augmented generation.")
]

# Stream the response
for chunk in chat.stream(messages):
    # Process each chunk as it arrives
    print(chunk.message.content, end="", flush=True)
```

#### Multi-turn Conversations

You can have multi-turn conversations with chat history:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_glean import ChatGlean

# Initialize the chat model with chat saving enabled
chat = ChatGlean(save_chat=True)

# Start a conversation
conversation = [
    SystemMessage(content="You are a helpful AI assistant for our company.")
]

# First turn
conversation.append(HumanMessage(content="What are our main projects?"))
response = chat.invoke(conversation)
print(f"AI: {response.content}")
conversation.append(response)

# Second turn
conversation.append(HumanMessage(content="Which one has the highest priority?"))
response = chat.invoke(conversation)
print(f"AI: {response.content}")

# The chat_id is saved in the chat model
print(f"Chat ID: {chat.chat_id}")
```

#### Chat with RAG

You can combine the chat model with a retriever for RAG:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_glean import ChatGlean, GleanSearchRetriever

# Initialize components
retriever = GleanSearchRetriever()
chat = ChatGlean()

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the retrieved information: {context}"),
    ("human", "{question}")
])

# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create a RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

# Run the chain
response = rag_chain.invoke("What are our company policies?")
print(response)
```

### Using the Retriever

The `GleanSearchRetriever` allows you to search and retrieve documents from Glean:

```python
from langchain_glean.retrievers import GleanSearchRetriever

# Initialize the retriever (will use environment variables)
retriever = GleanSearchRetriever()

# Search for documents
documents = retriever.invoke("quarterly sales report")

# Process the results
for doc in documents:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"URL: {doc.metadata.get('url')}")
    print(f"Content: {doc.page_content}")
    print("---")
```

### Using the Tool

The `GleanSearchTool` can be used in LangChain agents to search Glean:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool

# Initialize the retriever (will use environment variables)
retriever = GleanSearchRetriever()

# Create the tool
glean_tool = GleanSearchTool(
    retriever=retriever,
    name="glean_search",
    description="Search for information in your organization's content using Glean."
)

# Create an agent with the tool
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to Glean search."),
    ("user", "{input}")
])

agent = create_openai_tools_agent(llm, [glean_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[glean_tool])

# Run the agent
response = agent_executor.invoke({"input": "Find the latest quarterly report"})
print(response["output"])
```

### Integration with LangChain Chains

You can integrate the retriever with LangChain chains for more complex workflows:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_glean.retrievers import GleanSearchRetriever

# Initialize the retriever (will use environment variables)
retriever = GleanSearchRetriever()

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the chain
result = chain.invoke("What were our Q2 sales results?")
print(result)
```

## Advanced Usage

### Chat Model Parameters

You can customize the chat model behavior with additional parameters:

```python
from langchain_glean import ChatGlean

# Initialize with custom parameters
chat = ChatGlean(
    save_chat=True,  # Save the chat session in Glean
    chat_id="existing-chat-id",  # Continue an existing chat
    agent="GPT",  # Specify the agent type (DEFAULT, GPT, etc.)
    mode="SEARCH",  # Specify the mode (DEFAULT, SEARCH, etc.)
    timeout=30  # Timeout in seconds for API requests
)
```

### Search Parameters

You can customize your search by passing additional parameters:

```python
# Search with additional parameters
documents = retriever.invoke(
    "quarterly sales report",
    page_size=5,  # Number of results to return
    disable_spellcheck=True,  # Disable spellcheck
    max_snippet_size=200  # Maximum snippet size
)
```

## Contributing

For information on setting up a development environment and contributing to the project, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE)
