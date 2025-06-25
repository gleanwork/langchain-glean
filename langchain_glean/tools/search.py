from typing import Any, Dict, Union

from glean.api_client import errors
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.retrievers.search import SearchBasicRequest


class GleanSearchTool(BaseTool):
    """Tool for searching Glean using the GleanSearchRetriever."""

    name: str = "glean_search"
    description: str = """
    Search for information in Glean.
    Useful for finding documents, emails, messages, and other content across connected datasources.
    Input should be a search query or a JSON object with search parameters.
    """

    retriever: GleanSearchRetriever = Field(..., description="The GleanSearchRetriever to use for searching")
    return_direct: bool = False

    args_schema: type = SearchBasicRequest

    def _run(self, query: Union[str, Dict[str, Any], SearchBasicRequest]) -> str:
        """Run the tool.

        Args:
            query: Either a string query or a dictionary of search parameters

        Returns:
            A formatted string with the search results
        """
        try:
            if isinstance(query, (str, SearchBasicRequest)):
                docs = self.retriever.invoke(query)  # type: ignore[arg-type]
            else:
                query_str = query.pop("query", "")
                docs = self.retriever.invoke(query_str, **query)

            if not docs:
                return "No results found."

            results_str = []
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", "Untitled")
                url = doc.metadata.get("url", "No URL")
                source = doc.metadata.get("datasource", "Unknown Source")
                results_str.append(f"Result {i + 1}: {title} ({source})")
                results_str.append(f"URL: {url}")
                results_str.append(f"Content: {doc.page_content}")
                results_str.append("")

            return "\n".join(results_str)

        except errors.GleanError as e:
            error_details = f"Glean API error: {str(e)}"
            if hasattr(e, "raw_response") and e.raw_response:
                error_details += f": {e.raw_response}"
            return error_details
        except Exception as e:
            return f"Error running Glean search: {str(e)}"

    async def _arun(self, query: Union[str, Dict[str, Any], SearchBasicRequest]) -> str:
        """Run the tool asynchronously.

        Args:
            query: Either a string query or a dictionary of search parameters

        Returns:
            A formatted string with the search results
        """
        try:
            if isinstance(query, (str, SearchBasicRequest)):
                docs = await self.retriever.ainvoke(query)  # type: ignore[arg-type]
            else:
                query_str = query.pop("query", "")
                docs = await self.retriever.ainvoke(query_str, **query)

            if not docs:
                return "No results found."

            results_str = []
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", "Untitled")
                url = doc.metadata.get("url", "No URL")
                source = doc.metadata.get("datasource", "Unknown Source")
                results_str.append(f"Result {i + 1}: {title} ({source})")
                results_str.append(f"URL: {url}")
                results_str.append(f"Content: {doc.page_content}")
                results_str.append("")

            return "\n".join(results_str)

        except errors.GleanError as e:
            error_details = f"Glean API error: {str(e)}"
            if hasattr(e, "raw_response") and e.raw_response:
                error_details += f": {e.raw_response}"
            return error_details
        except Exception as e:
            return f"Error running Glean search: {str(e)}"
