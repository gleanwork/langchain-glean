from importlib import metadata

from langchain_glean.chat_models import ChatGlean
from langchain_glean.retrievers import GleanSearchParameters, GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatGlean",
    "GleanSearchParameters",
    "GleanSearchRetriever",
    "GleanSearchTool",
    "__version__",
]
