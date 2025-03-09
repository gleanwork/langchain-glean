from typing import Any, Dict, Union

from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_glean.client.glean_client import GleanClientError, GleanConnectionError, GleanHTTPError
from langchain_glean.retrievers import GleanSearchRetriever


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

    def _run(self, query: Union[str, Dict[str, Any]]) -> str:
        """Run the tool.

        Args:
            query: Either a string query or a dictionary of search parameters

        Returns:
            A formatted string with the search results
        """
        try:
            search_kwargs: Dict[str, Any] = {}

            if isinstance(query, str):
                query_text = query
            else:
                if "query" not in query:
                    return "Error: Search query is required"

                query_text = query.pop("query")

                for key, value in query.items():
                    if key in [
                        "page_size",
                        "cursor",
                        "disable_spellcheck",
                        "max_snippet_size",
                        "result_tab_ids",
                        "timeout_millis",
                        "tracking_token",
                        "request_options",
                    ]:
                        search_kwargs[key] = value

            try:
                docs = self.retriever.invoke(query_text, **search_kwargs)
            except GleanHTTPError as http_err:
                error_details = f"HTTP Error {http_err.status_code}"
                if http_err.response:
                    error_details += f": {http_err.response}"
                return f"Error searching Glean: {error_details}"
            except GleanConnectionError as conn_err:
                return f"Error connecting to Glean: {str(conn_err)}"
            except GleanClientError as client_err:
                return f"Glean client error: {str(client_err)}"

            if not docs:
                return "No results found."

            results = []
            for i, doc in enumerate(docs, 1):
                result = f"Result {i}:\n"
                result += f"Title: {doc.metadata.get('title', 'No title')}\n"
                result += f"URL: {doc.metadata.get('url', 'No URL')}\n"
                result += f"Content: {doc.page_content}\n"
                results.append(result)

            return "\n\n".join(results)

        except Exception as e:
            return f"Error searching Glean: {str(e)}"

    async def _arun(self, query: Union[str, Dict[str, Any]]) -> str:
        """Run the tool asynchronously.

        Args:
            query: Either a string query or a dictionary of search parameters

        Returns:
            A formatted string with the search results
        """
        try:
            search_kwargs: Dict[str, Any] = {}

            if isinstance(query, str):
                query_text = query
            else:
                if "query" not in query:
                    return "Error: Search query is required"

                query_text = query.pop("query")

                for key, value in query.items():
                    if key in [
                        "page_size",
                        "cursor",
                        "disable_spellcheck",
                        "max_snippet_size",
                        "result_tab_ids",
                        "timeout_millis",
                        "tracking_token",
                        "request_options",
                    ]:
                        search_kwargs[key] = value

            try:
                docs = await self.retriever.ainvoke(query_text, **search_kwargs)
            except GleanHTTPError as http_err:
                error_details = f"HTTP Error {http_err.status_code}"
                if http_err.response:
                    error_details += f": {http_err.response}"
                return f"Error searching Glean: {error_details}"
            except GleanConnectionError as conn_err:
                return f"Error connecting to Glean: {str(conn_err)}"
            except GleanClientError as client_err:
                return f"Glean client error: {str(client_err)}"

            if not docs:
                return "No results found."

            results = []
            for i, doc in enumerate(docs, 1):
                result = f"Result {i}:\n"
                result += f"Title: {doc.metadata.get('title', 'No title')}\n"
                result += f"URL: {doc.metadata.get('url', 'No URL')}\n"
                result += f"Content: {doc.page_content}\n"
                results.append(result)

            return "\n\n".join(results)

        except Exception as e:
            return f"Error searching Glean: {str(e)}"
