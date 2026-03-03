import os
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_glean.retrievers.search import GleanSearchRetriever, SearchBasicRequest
from langchain_glean.tools.search import GleanSearchTool


class TestGleanSearchTool:
    """Test the GleanSearchTool class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        # Create mock retriever
        self.mock_retriever = MagicMock(spec=GleanSearchRetriever)

        # Set up sample documents for retriever responses
        self.sample_doc1 = Document(
            page_content="Glean is an enterprise search platform.",
            metadata={
                "title": "About Glean",
                "url": "https://example.com/about-glean",
                "datasource": "Confluence",
            },
        )

        self.sample_doc2 = Document(
            page_content="Glean uses AI to improve search results.",
            metadata={
                "title": "Glean AI Features",
                "url": "https://example.com/glean-ai",
                "datasource": "Google Drive",
            },
        )

        # Set up mock responses
        self.mock_retriever.invoke.return_value = [self.sample_doc1, self.sample_doc2]
        self.mock_retriever.ainvoke.return_value = [self.sample_doc1, self.sample_doc2]

        # Initialize the tool with the mock retriever
        self.tool = GleanSearchTool(retriever=self.mock_retriever)

        yield

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        assert self.tool.name == "glean_search"
        assert "Search for information in Glean" in self.tool.description
        assert self.tool.retriever == self.mock_retriever
        assert self.tool.args_schema == SearchBasicRequest
        assert not self.tool.return_direct

    def test_run_with_string_query(self) -> None:
        """Test _run with a plain string query."""
        result = self.tool._run("enterprise search")

        self.mock_retriever.invoke.assert_called_once_with("enterprise search")

        assert "Result 1: About Glean (Confluence)" in result
        assert "URL: https://example.com/about-glean" in result
        assert "Content: Glean is an enterprise search platform." in result
        assert "Result 2: Glean AI Features (Google Drive)" in result
        assert "URL: https://example.com/glean-ai" in result
        assert "Content: Glean uses AI to improve search results." in result

    def test_run_with_search_basic_request(self) -> None:
        """Test _run with a SearchBasicRequest input."""
        request = SearchBasicRequest(query="enterprise search")

        result = self.tool._run(request)

        self.mock_retriever.invoke.assert_called_once_with(request)

        assert "Result 1: About Glean (Confluence)" in result
        assert "Result 2: Glean AI Features (Google Drive)" in result

    def test_run_with_dict_query(self) -> None:
        """Test _run with a dictionary input (pops 'query' key, passes rest as kwargs)."""
        result = self.tool._run({"query": "enterprise search", "page_size": 5})

        self.mock_retriever.invoke.assert_called_once_with("enterprise search", page_size=5)

        assert "Result 1: About Glean (Confluence)" in result

    def test_run_with_dict_no_query_key(self) -> None:
        """Test _run with a dict that has no 'query' key defaults to empty string."""
        result = self.tool._run({"page_size": 5})

        self.mock_retriever.invoke.assert_called_once_with("", page_size=5)

        assert "Result 1: About Glean (Confluence)" in result

    def test_run_with_no_results(self) -> None:
        """Test _run when no results are found."""
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("nonexistent topic")

        assert result == "No results found."

    def test_run_with_missing_metadata(self) -> None:
        """Test _run with documents that have missing metadata fields."""
        doc = Document(page_content="Some content", metadata={})
        self.mock_retriever.invoke.return_value = [doc]

        result = self.tool._run("test query")

        assert "Result 1: Untitled (Unknown Source)" in result
        assert "URL: No URL" in result
        assert "Content: Some content" in result

    def test_run_with_glean_error(self) -> None:
        """Test _run when a GleanError occurs."""
        from glean.api_client import errors

        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("Search failed", raw_response=mock_response)
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("test query")

        assert "Glean API error" in result
        assert "Search failed" in result

    def test_run_with_glean_error_no_raw_response_text(self) -> None:
        """Test _run when a GleanError occurs with empty raw_response text."""
        from glean.api_client import errors

        mock_response = MagicMock()
        mock_response.text = ""
        error = errors.GleanError("Search failed", raw_response=mock_response)
        # Ensure raw_response is falsy for the hasattr+truthy check
        error.raw_response = None
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("test query")

        assert "Glean API error" in result
        assert "Search failed" in result

    def test_run_with_generic_exception(self) -> None:
        """Test _run when a generic exception occurs."""
        self.mock_retriever.invoke.side_effect = Exception("Something went wrong")

        result = self.tool._run("test query")

        assert "Error running Glean search" in result
        assert "Something went wrong" in result

    async def test_arun_with_string_query(self) -> None:
        """Test _arun with a plain string query."""
        result = await self.tool._arun("enterprise search")

        self.mock_retriever.ainvoke.assert_called_once_with("enterprise search")

        assert "Result 1: About Glean (Confluence)" in result
        assert "URL: https://example.com/about-glean" in result
        assert "Content: Glean is an enterprise search platform." in result
        assert "Result 2: Glean AI Features (Google Drive)" in result

    async def test_arun_with_search_basic_request(self) -> None:
        """Test _arun with a SearchBasicRequest input."""
        request = SearchBasicRequest(query="enterprise search")

        result = await self.tool._arun(request)

        self.mock_retriever.ainvoke.assert_called_once_with(request)

        assert "Result 1: About Glean (Confluence)" in result
        assert "Result 2: Glean AI Features (Google Drive)" in result

    async def test_arun_with_dict_query(self) -> None:
        """Test _arun with a dictionary input."""
        result = await self.tool._arun({"query": "enterprise search", "page_size": 5})

        self.mock_retriever.ainvoke.assert_called_once_with("enterprise search", page_size=5)

        assert "Result 1: About Glean (Confluence)" in result

    async def test_arun_with_no_results(self) -> None:
        """Test _arun when no results are found."""
        self.mock_retriever.ainvoke.return_value = []

        result = await self.tool._arun("nonexistent topic")

        assert result == "No results found."

    async def test_arun_with_glean_error(self) -> None:
        """Test _arun when a GleanError occurs."""
        from glean.api_client import errors

        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("Search failed", raw_response=mock_response)
        self.mock_retriever.ainvoke.side_effect = error

        result = await self.tool._arun("test query")

        assert "Glean API error" in result
        assert "Search failed" in result

    async def test_arun_with_generic_exception(self) -> None:
        """Test _arun when a generic exception occurs."""
        self.mock_retriever.ainvoke.side_effect = Exception("Something went wrong")

        result = await self.tool._arun("test query")

        assert "Error running Glean search" in result
        assert "Something went wrong" in result
