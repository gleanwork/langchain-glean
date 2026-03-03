import os
from unittest.mock import MagicMock, patch

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
            page_content="This is a sample snippet about Glean search.",
            metadata={
                "title": "Glean Search Documentation",
                "url": "https://example.com/doc1",
                "datasource": "confluence",
                "doc_type": "Article",
            },
        )

        self.sample_doc2 = Document(
            page_content="Another document about enterprise search.",
            metadata={
                "title": "Enterprise Search Guide",
                "url": "https://example.com/doc2",
                "datasource": "gdrive",
                "doc_type": "Document",
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
        """Test _run with a simple string query."""
        result = self.tool._run("search query")

        # Verify the retriever's invoke method was called
        self.mock_retriever.invoke.assert_called_once_with("search query")

        # Check the output format
        assert "Result 1: Glean Search Documentation (confluence)" in result
        assert "URL: https://example.com/doc1" in result
        assert "Content: This is a sample snippet about Glean search." in result
        assert "Result 2: Enterprise Search Guide (gdrive)" in result
        assert "URL: https://example.com/doc2" in result
        assert "Content: Another document about enterprise search." in result

    def test_run_with_search_basic_request(self) -> None:
        """Test _run with a SearchBasicRequest."""
        request = SearchBasicRequest(query="test query", data_sources=["confluence"])

        result = self.tool._run(request)

        # Verify the retriever's invoke method was called with the request
        self.mock_retriever.invoke.assert_called_once_with(request)

        # Check the output format
        assert "Result 1: Glean Search Documentation (confluence)" in result

    def test_run_with_dict_query(self) -> None:
        """Test _run with a dict input (query + extra kwargs)."""
        result = self.tool._run({"query": "test query", "page_size": 5})

        # Verify the retriever's invoke method was called with the query string and extra kwargs
        self.mock_retriever.invoke.assert_called_once_with("test query", page_size=5)

        assert "Result 1: Glean Search Documentation (confluence)" in result

    def test_run_with_no_results(self) -> None:
        """Test _run when no results are found."""
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("nonexistent query")

        assert result == "No results found."

    def test_run_with_missing_metadata(self) -> None:
        """Test _run with documents that have missing metadata."""
        doc_no_metadata = Document(
            page_content="Content with no metadata",
            metadata={},
        )
        self.mock_retriever.invoke.return_value = [doc_no_metadata]

        result = self.tool._run("test")

        assert "Result 1: Untitled (Unknown Source)" in result
        assert "URL: No URL" in result
        assert "Content: Content with no metadata" in result

    def test_run_with_glean_error(self) -> None:
        """Test _run when a GleanError occurs."""
        from glean.api_client import errors

        mock_response = MagicMock()
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("test query")

        assert "Glean API error" in result
        assert "Test error" in result

    def test_run_with_generic_exception(self) -> None:
        """Test _run when a generic exception occurs."""
        self.mock_retriever.invoke.side_effect = Exception("Generic error")

        result = self.tool._run("test query")

        assert "Error running Glean search" in result
        assert "Generic error" in result

    async def test_arun_with_string_query(self) -> None:
        """Test _arun with a simple string query."""
        result = await self.tool._arun("search query")

        # Verify the retriever's ainvoke method was called
        self.mock_retriever.ainvoke.assert_called_once_with("search query")

        assert "Result 1: Glean Search Documentation (confluence)" in result
        assert "Result 2: Enterprise Search Guide (gdrive)" in result

    async def test_arun_with_search_basic_request(self) -> None:
        """Test _arun with a SearchBasicRequest."""
        request = SearchBasicRequest(query="test query", data_sources=["confluence"])

        result = await self.tool._arun(request)

        # Verify the retriever's ainvoke method was called with the request
        self.mock_retriever.ainvoke.assert_called_once_with(request)

        assert "Result 1: Glean Search Documentation (confluence)" in result

    async def test_arun_with_dict_query(self) -> None:
        """Test _arun with a dict input (query + extra kwargs)."""
        result = await self.tool._arun({"query": "test query", "page_size": 5})

        # Verify the retriever's ainvoke method was called with the query string and extra kwargs
        self.mock_retriever.ainvoke.assert_called_once_with("test query", page_size=5)

        assert "Result 1: Glean Search Documentation (confluence)" in result

    async def test_arun_with_no_results(self) -> None:
        """Test _arun when no results are found."""
        self.mock_retriever.ainvoke.return_value = []

        result = await self.tool._arun("nonexistent query")

        assert result == "No results found."

    async def test_arun_with_glean_error(self) -> None:
        """Test _arun when a GleanError occurs."""
        from glean.api_client import errors

        mock_response = MagicMock()
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_retriever.ainvoke.side_effect = error

        result = await self.tool._arun("test query")

        assert "Glean API error" in result
        assert "Test error" in result

    async def test_arun_with_generic_exception(self) -> None:
        """Test _arun when a generic exception occurs."""
        self.mock_retriever.ainvoke.side_effect = Exception("Generic error")

        result = await self.tool._arun("test query")

        assert "Error running Glean search" in result
        assert "Generic error" in result
