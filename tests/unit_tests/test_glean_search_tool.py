from unittest.mock import MagicMock

import pytest
from glean.api_client import errors
from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.retrievers.search import SearchBasicRequest
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
            page_content="This is the first document content.",
            metadata={
                "title": "First Document",
                "url": "https://example.com/doc1",
                "datasource": "confluence",
            },
        )

        self.sample_doc2 = Document(
            page_content="This is the second document content.",
            metadata={
                "title": "Second Document",
                "url": "https://example.com/doc2",
                "datasource": "slack",
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

    # ===== SYNC _run TESTS =====

    def test_run_with_string_query(self) -> None:
        """Test _run with a plain string query."""
        result = self.tool._run("test query")

        self.mock_retriever.invoke.assert_called_once_with("test query")

        assert "Result 1: First Document (confluence)" in result
        assert "URL: https://example.com/doc1" in result
        assert "Content: This is the first document content." in result
        assert "Result 2: Second Document (slack)" in result
        assert "URL: https://example.com/doc2" in result
        assert "Content: This is the second document content." in result

    def test_run_with_search_basic_request(self) -> None:
        """Test _run with a SearchBasicRequest object."""
        request = SearchBasicRequest(query="typed query")

        result = self.tool._run(request)

        self.mock_retriever.invoke.assert_called_once_with(request)

        assert "Result 1: First Document (confluence)" in result

    def test_run_with_dict_query(self) -> None:
        """Test _run with a dictionary input (pops 'query' key)."""
        result = self.tool._run({"query": "dict query", "extra_param": "value"})

        self.mock_retriever.invoke.assert_called_once_with("dict query", extra_param="value")

        assert "Result 1: First Document (confluence)" in result

    def test_run_with_dict_missing_query_key(self) -> None:
        """Test _run with a dict that has no 'query' key defaults to empty string."""
        result = self.tool._run({"extra_param": "value"})

        self.mock_retriever.invoke.assert_called_once_with("", extra_param="value")

        assert "Result 1: First Document (confluence)" in result

    def test_run_with_no_results(self) -> None:
        """Test _run when no results are returned."""
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("no results query")

        assert result == "No results found."

    def test_run_with_missing_metadata(self) -> None:
        """Test _run handles documents with missing metadata fields."""
        doc_no_metadata = Document(
            page_content="Content with no metadata.",
            metadata={},
        )
        self.mock_retriever.invoke.return_value = [doc_no_metadata]

        result = self.tool._run("sparse query")

        assert "Result 1: Untitled (Unknown Source)" in result
        assert "URL: No URL" in result
        assert "Content: Content with no metadata." in result

    def test_run_with_glean_error(self) -> None:
        """Test _run when a GleanError occurs."""
        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("error query")

        assert "Glean API error" in result
        assert "Test error" in result

    def test_run_with_glean_error_no_raw_response(self) -> None:
        """Test _run when a GleanError occurs with empty raw_response."""
        mock_response = MagicMock()
        mock_response.text = ""
        error = errors.GleanError("Test error", raw_response=mock_response)
        # Remove raw_response so hasattr check returns False
        error.raw_response = None
        self.mock_retriever.invoke.side_effect = error

        result = self.tool._run("error query")

        assert "Glean API error: Test error" in result

    def test_run_with_generic_exception(self) -> None:
        """Test _run when a generic exception occurs."""
        self.mock_retriever.invoke.side_effect = Exception("Generic error")

        result = self.tool._run("error query")

        assert "Error running Glean search: Generic error" in result

    # ===== ASYNC _arun TESTS =====

    async def test_arun_with_string_query(self) -> None:
        """Test _arun with a plain string query."""
        result = await self.tool._arun("test query")

        self.mock_retriever.ainvoke.assert_called_once_with("test query")

        assert "Result 1: First Document (confluence)" in result
        assert "URL: https://example.com/doc1" in result
        assert "Content: This is the first document content." in result
        assert "Result 2: Second Document (slack)" in result

    async def test_arun_with_search_basic_request(self) -> None:
        """Test _arun with a SearchBasicRequest object."""
        request = SearchBasicRequest(query="typed query")

        result = await self.tool._arun(request)

        self.mock_retriever.ainvoke.assert_called_once_with(request)

        assert "Result 1: First Document (confluence)" in result

    async def test_arun_with_dict_query(self) -> None:
        """Test _arun with a dictionary input."""
        result = await self.tool._arun({"query": "dict query", "extra_param": "value"})

        self.mock_retriever.ainvoke.assert_called_once_with("dict query", extra_param="value")

        assert "Result 1: First Document (confluence)" in result

    async def test_arun_with_no_results(self) -> None:
        """Test _arun when no results are returned."""
        self.mock_retriever.ainvoke.return_value = []

        result = await self.tool._arun("no results query")

        assert result == "No results found."

    async def test_arun_with_glean_error(self) -> None:
        """Test _arun when a GleanError occurs."""
        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_retriever.ainvoke.side_effect = error

        result = await self.tool._arun("error query")

        assert "Glean API error" in result
        assert "Test error" in result

    async def test_arun_with_generic_exception(self) -> None:
        """Test _arun when a generic exception occurs."""
        self.mock_retriever.ainvoke.side_effect = Exception("Generic error")

        result = await self.tool._arun("error query")

        assert "Error running Glean search: Generic error" in result
