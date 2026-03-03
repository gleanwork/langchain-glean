from unittest.mock import MagicMock

import pytest
from glean.api_client import errors
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
                "datasource": "confluence",
            },
        )

        self.sample_doc2 = Document(
            page_content="How to set up Glean API tokens.",
            metadata={
                "title": "API Setup Guide",
                "url": "https://example.com/api-guide",
                "datasource": "gdrive",
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
        result = self.tool._run("enterprise search")

        self.mock_retriever.invoke.assert_called_once_with("enterprise search")

        assert "Result 1: About Glean (confluence)" in result
        assert "URL: https://example.com/about-glean" in result
        assert "Content: Glean is an enterprise search platform." in result
        assert "Result 2: API Setup Guide (gdrive)" in result
        assert "URL: https://example.com/api-guide" in result
        assert "Content: How to set up Glean API tokens." in result

    def test_run_with_search_basic_request(self) -> None:
        """Test _run with a SearchBasicRequest object."""
        request = SearchBasicRequest(query="enterprise search", data_sources=["confluence"])

        result = self.tool._run(request)

        self.mock_retriever.invoke.assert_called_once_with(request)

        assert "Result 1: About Glean (confluence)" in result

    def test_run_with_dict_query(self) -> None:
        """Test _run with a dictionary query (pops 'query' key and passes rest as kwargs)."""
        result = self.tool._run({"query": "enterprise search", "k": 5})

        self.mock_retriever.invoke.assert_called_once_with("enterprise search", k=5)

        assert "Result 1: About Glean (confluence)" in result

    def test_run_with_dict_missing_query_key(self) -> None:
        """Test _run with a dict that has no 'query' key defaults to empty string."""
        result = self.tool._run({"k": 5})

        self.mock_retriever.invoke.assert_called_once_with("", k=5)

        assert "Result 1: About Glean (confluence)" in result

    def test_run_with_no_results(self) -> None:
        """Test _run when no results are found."""
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("nonexistent topic")

        assert result == "No results found."

    def test_run_with_missing_metadata_fields(self) -> None:
        """Test _run with documents that have missing metadata fields."""
        doc_missing_metadata = Document(
            page_content="Some content",
            metadata={},
        )
        self.mock_retriever.invoke.return_value = [doc_missing_metadata]

        result = self.tool._run("test query")

        assert "Result 1: Untitled (Unknown Source)" in result
        assert "URL: No URL" in result
        assert "Content: Some content" in result

    def test_run_with_glean_error(self) -> None:
        """Test _run when a GleanError occurs."""
        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("API rate limit exceeded", raw_response=mock_response)
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("enterprise search")

        assert "Glean API error" in result
        assert "API rate limit exceeded" in result

    def test_run_with_generic_exception(self) -> None:
        """Test _run when a generic exception occurs."""
        self.mock_retriever.invoke.side_effect = Exception("Connection timeout")

        result = self.tool._run("enterprise search")

        assert "Error running Glean search" in result
        assert "Connection timeout" in result

    @pytest.mark.asyncio
    async def test_arun_with_string_query(self) -> None:
        """Test _arun with a simple string query."""
        result = await self.tool._arun("enterprise search")

        self.mock_retriever.ainvoke.assert_called_once_with("enterprise search")

        assert "Result 1: About Glean (confluence)" in result
        assert "Result 2: API Setup Guide (gdrive)" in result

    @pytest.mark.asyncio
    async def test_arun_with_search_basic_request(self) -> None:
        """Test _arun with a SearchBasicRequest object."""
        request = SearchBasicRequest(query="enterprise search", data_sources=["confluence"])

        result = await self.tool._arun(request)

        self.mock_retriever.ainvoke.assert_called_once_with(request)

        assert "Result 1: About Glean (confluence)" in result

    @pytest.mark.asyncio
    async def test_arun_with_dict_query(self) -> None:
        """Test _arun with a dictionary query."""
        result = await self.tool._arun({"query": "enterprise search", "k": 5})

        self.mock_retriever.ainvoke.assert_called_once_with("enterprise search", k=5)

        assert "Result 1: About Glean (confluence)" in result

    @pytest.mark.asyncio
    async def test_arun_with_no_results(self) -> None:
        """Test _arun when no results are found."""
        self.mock_retriever.ainvoke.return_value = []

        result = await self.tool._arun("nonexistent topic")

        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_arun_with_glean_error(self) -> None:
        """Test _arun when a GleanError occurs."""
        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("API rate limit exceeded", raw_response=mock_response)
        self.mock_retriever.ainvoke.side_effect = error

        result = await self.tool._arun("enterprise search")

        assert "Glean API error" in result
        assert "API rate limit exceeded" in result

    @pytest.mark.asyncio
    async def test_arun_with_generic_exception(self) -> None:
        """Test _arun when a generic exception occurs."""
        self.mock_retriever.ainvoke.side_effect = Exception("Connection timeout")

        result = await self.tool._arun("enterprise search")

        assert "Error running Glean search" in result
        assert "Connection timeout" in result
