# langchain-glean

This package contains the LangChain integration with Glean

## Installation

```bash
pip install -U langchain-glean
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatGlean` class exposes chat models from Glean.

```python
from langchain_glean import ChatGlean

llm = ChatGlean()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`GleanEmbeddings` class exposes embeddings from Glean.

```python
from langchain_glean import GleanEmbeddings

embeddings = GleanEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`GleanLLM` class exposes LLMs from Glean.

```python
from langchain_glean import GleanLLM

llm = GleanLLM()
llm.invoke("The meaning of life is")
```
