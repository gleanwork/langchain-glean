from importlib import metadata

from langchain_glean.chat_models import ChatGlean
from langchain_glean.document_loaders import GleanLoader
from langchain_glean.embeddings import GleanEmbeddings
from langchain_glean.retrievers import GleanRetriever
from langchain_glean.toolkits import GleanToolkit
from langchain_glean.tools import GleanTool
from langchain_glean.vectorstores import GleanVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatGlean",
    "GleanVectorStore",
    "GleanEmbeddings",
    "GleanLoader",
    "GleanRetriever",
    "GleanToolkit",
    "GleanTool",
    "__version__",
]
