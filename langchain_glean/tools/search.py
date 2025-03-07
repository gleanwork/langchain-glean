from typing import Any, Dict, Union

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

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
            if isinstance(query, str):
                search_params = {"query": query}
            else:
                search_params = query

            if "query" not in search_params:
                return "Error: Search query is required"

            query_text = search_params.pop("query")

            docs = self.retriever.get_relevant_documents(query_text, **search_params)

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
            if isinstance(query, str):
                search_params = {"query": query}
            else:
                search_params = query

            if "query" not in search_params:
                return "Error: Search query is required"

            query_text = search_params.pop("query")

            docs = await self.retriever.aget_relevant_documents(query_text, **search_params)

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
