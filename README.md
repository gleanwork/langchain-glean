# langchain-glean

This package contains the LangChain integration with Glean

## Installation

```bash
pip install -U langchain-glean
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [go-task](https://taskfile.dev/) for task running.

### Setup Development Environment

1. Install uv:

```bash
pip install uv
```

2. Install go-task:

```bash
brew install go-task
```

3. Set up the development environment:

```bash
task setup
```

This will create a virtual environment and install all dependencies.

4. Activate the virtual environment:

```bash
source .venv/bin/activate
```

### Running Tests

```bash
task test
```

### Running Linters

```bash
task lint
```

### Migrating from Poetry

If you were previously using Poetry, you can migrate to uv by running:

```bash
./scripts/migrate_to_uv.sh
```

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
