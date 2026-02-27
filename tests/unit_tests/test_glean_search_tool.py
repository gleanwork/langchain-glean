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
        self.mock_retriever = MagicMock(spec=GleanSearchRetriever)

        self.sample_doc1 = Document(
            page_content="Glean is an enterprise search platform.",
            metadata={
                "title": "About Glean",
                "url": "https://example.com/about",
                "datasource": "confluence",
                "doc_type": "Article",
            },
        )

        self.sample_doc2 = Document(
            page_content="Glean connects all your work apps.",
            metadata={
                "title": "Glean Features",
                "url": "https://example.com/features",
                "datasource": "drive",
                "doc_type": "Document",
            },
        )

        self.mock_retriever.invoke.return_value = [self.sample_doc1, self.sample_doc2]
        self.mock_retriever.ainvoke.return_value = [self.sample_doc1, self.sample_doc2]

        self.tool = GleanSearchTool(retriever=self.mock_retriever)

        yield

    # ===== Initialization =====

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        assert self.tool.name == "glean_search"
        assert "Search for information in Glean" in self.tool.description
        assert self.tool.retriever == self.mock_retriever
        assert self.tool.args_schema == SearchBasicRequest
        assert not self.tool.return_direct

    # ===== _run tests =====

    def test_run_with_string_query(self) -> None:
        """Test _run with a simple string query."""
        result = self.tool._run("enterprise search")

        self.mock_retriever.invoke.assert_called_once_with("enterprise search")

        assert "Result 1: About Glean (confluence)" in result
        assert "URL: https://example.com/about" in result
        assert "Content: Glean is an enterprise search platform." in result
        assert "Result 2: Glean Features (drive)" in result

    def test_run_with_search_basic_request(self) -> None:
        """Test _run with a SearchBasicRequest object."""
        request = SearchBasicRequest(query="enterprise search")

        result = self.tool._run(request)

        self.mock_retriever.invoke.assert_called_once_with(request)

        assert "Result 1: About Glean" in result
        assert "Result 2: Glean Features" in result

    def test_run_with_dict_query(self) -> None:
        """Test _run with a dict input (query + extra kwargs)."""
        result = self.tool._run({"query": "search platform", "page_size": 5})

        self.mock_retriever.invoke.assert_called_once_with(
            "search platform", page_size=5
        )

        assert "Result 1: About Glean" in result

    def test_run_with_dict_missing_query_key(self) -> None:
        """Test _run with dict that has no 'query' key defaults to empty string."""
        result = self.tool._run({"page_size": 5})

        self.mock_retriever.invoke.assert_called_once_with("", page_size=5)

        assert "Result 1: About Glean" in result

    def test_run_with_no_results(self) -> None:
        """Test _run when no results are returned."""
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("nonexistent topic")

        assert result == "No results found."

    def test_run_with_missing_metadata(self) -> None:
        """Test _run with documents that have missing metadata fields."""
        doc = Document(page_content="Some content", metadata={})
        self.mock_retriever.invoke.return_value = [doc]

        result = self.tool._run("test")

        assert "Result 1: Untitled (Unknown Source)" in result
        assert "URL: No URL" in result
        assert "Content: Some content" in result

    def test_run_with_glean_error(self) -> None:
        """Test _run when a GleanError occurs."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        error = errors.GleanError("API rate limit exceeded", raw_response=mock_response)
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("test")

        assert "Glean API error:" in result
        assert "API rate limit exceeded" in result

    def test_run_with_glean_error_and_raw_response(self) -> None:
        """Test _run when a GleanError with a truthy raw_response occurs."""
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: '{"error": "invalid query"}'
        mock_response.__bool__ = lambda self: True
        error = errors.GleanError("Bad request", raw_response=mock_response)
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("test")

        assert "Glean API error:" in result

    def test_run_with_generic_error(self) -> None:
        """Test _run when a generic exception occurs."""
        self.mock_retriever.invoke.side_effect = Exception("Connection timeout")

        result = self.tool._run("test")

        assert "Error running Glean search: Connection timeout" in result

    # ===== _arun tests =====

    async def test_arun_with_string_query(self) -> None:
        """Test _arun with a simple string query."""
        result = await self.tool._arun("enterprise search")

        self.mock_retriever.ainvoke.assert_called_once_with("enterprise search")

        assert "Result 1: About Glean (confluence)" in result
        assert "Result 2: Glean Features (drive)" in result

    async def test_arun_with_search_basic_request(self) -> None:
        """Test _arun with a SearchBasicRequest object."""
        request = SearchBasicRequest(query="enterprise search")

        result = await self.tool._arun(request)

        self.mock_retriever.ainvoke.assert_called_once_with(request)

        assert "Result 1: About Glean" in result

    async def test_arun_with_dict_query(self) -> None:
        """Test _arun with a dict input."""
        result = await self.tool._arun({"query": "search", "page_size": 3})

        self.mock_retriever.ainvoke.assert_called_once_with("search", page_size=3)

        assert "Result 1: About Glean" in result

    async def test_arun_with_no_results(self) -> None:
        """Test _arun when no results are returned."""
        self.mock_retriever.ainvoke.return_value = []

        result = await self.tool._arun("nonexistent")

        assert result == "No results found."

    async def test_arun_with_glean_error(self) -> None:
        """Test _arun when a GleanError occurs."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = errors.GleanError("API error", raw_response=mock_response)
        self.mock_retriever.ainvoke.side_effect = error

        result = await self.tool._arun("test")

        assert "Glean API error:" in result

    async def test_arun_with_generic_error(self) -> None:
        """Test _arun when a generic exception occurs."""
        self.mock_retriever.ainvoke.side_effect = Exception("Async timeout")

        result = await self.tool._arun("test")

        assert "Error running Glean search: Async timeout" in result
