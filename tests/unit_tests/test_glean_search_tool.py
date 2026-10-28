from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.retrievers.search import SearchBasicRequest
from langchain_glean.tools.search import GleanSearchTool


class TestGleanSearchTool:
    """Test the GleanSearchTool class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        self.mock_retriever = MagicMock(spec=GleanSearchRetriever)

        self.sample_doc = Document(
            page_content="Revenue grew 15% in Q2.",
            metadata={
                "title": "Q2 Sales Report",
                "url": "https://example.com/q2",
                "datasource": "gdrive",
            },
        )

        self.mock_retriever.invoke.return_value = [self.sample_doc]
        self.mock_retriever.ainvoke.return_value = [self.sample_doc]

        self.tool = GleanSearchTool(retriever=self.mock_retriever)

        yield

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        assert self.tool.name == "glean_search"
        assert self.tool.retriever == self.mock_retriever
        assert self.tool.args_schema == SearchBasicRequest
        assert not self.tool.return_direct

    def test_run_with_string_query(self) -> None:
        """Test the _run method with a string query."""
        result = self.tool._run("quarterly report")

        self.mock_retriever.invoke.assert_called_once_with("quarterly report")
        assert "Q2 Sales Report" in result
        assert "https://example.com/q2" in result

    def test_run_with_typed_request(self) -> None:
        """Test the _run method with a SearchBasicRequest."""
        request = SearchBasicRequest(query="quarterly report", data_sources=["gdrive"])

        result = self.tool._run(request)

        self.mock_retriever.invoke.assert_called_once_with(request)
        assert "Q2 Sales Report" in result

    def test_run_with_data_sources(self) -> None:
        """Test that _run accepts data_sources as a separate argument.

        ``args_schema`` advertises both ``query`` and ``data_sources``, so
        LangChain unpacks the validated schema into keyword arguments.
        """
        result = self.tool._run(query="quarterly report", data_sources=["slack"])

        assert "Q2 Sales Report" in result
        self.mock_retriever.invoke.assert_called_once_with("quarterly report", data_sources=["slack"])

    def test_invoke_with_data_sources(self) -> None:
        """Test the full LangChain tool-call path with data_sources."""
        result = self.tool.invoke({"query": "quarterly report", "data_sources": ["slack"]})

        assert "Q2 Sales Report" in result
        self.mock_retriever.invoke.assert_called_once_with("quarterly report", data_sources=["slack"])

    def test_run_with_dict_query(self) -> None:
        """Test the _run method with a dictionary of search parameters."""
        params = {"query": "quarterly report", "page_size": 5}

        result = self.tool._run(dict(params))

        assert "Q2 Sales Report" in result
        self.mock_retriever.invoke.assert_called_once_with("quarterly report", page_size=5)

    def test_run_with_dict_query_keeps_data_sources(self) -> None:
        """Test that a dictionary input keeps data_sources next to other parameters."""
        result = self.tool._run({"query": "quarterly report", "data_sources": ["slack"], "page_size": 5})

        assert "Q2 Sales Report" in result
        self.mock_retriever.invoke.assert_called_once_with("quarterly report", data_sources=["slack"], page_size=5)

    def test_run_does_not_mutate_dict_query(self) -> None:
        """Test that _run leaves a caller-owned dictionary untouched."""
        params = {"query": "quarterly report", "page_size": 5}

        self.tool._run(params)

        assert params == {"query": "quarterly report", "page_size": 5}

    def test_run_with_no_results(self) -> None:
        """Test the _run method when no results are found."""
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("nothing matches this")

        assert result == "No results found."

    def test_run_with_error(self) -> None:
        """Test the _run method when an error occurs."""
        self.mock_retriever.invoke.side_effect = Exception("Test error")

        result = self.tool._run("quarterly report")

        assert "Error running Glean search: Test error" in result

    async def test_arun_with_string_query(self) -> None:
        """Test the _arun method with a string query."""
        result = await self.tool._arun("quarterly report")

        self.mock_retriever.ainvoke.assert_called_once_with("quarterly report")
        assert "Q2 Sales Report" in result

    async def test_arun_with_typed_request(self) -> None:
        """Test the _arun method with a SearchBasicRequest."""
        request = SearchBasicRequest(query="quarterly report", data_sources=["gdrive"])

        result = await self.tool._arun(request)

        self.mock_retriever.ainvoke.assert_called_once_with(request)
        assert "Q2 Sales Report" in result

    async def test_arun_with_data_sources(self) -> None:
        """Test that _arun accepts data_sources as a separate argument."""
        result = await self.tool._arun(query="quarterly report", data_sources=["slack"])

        assert "Q2 Sales Report" in result
        self.mock_retriever.ainvoke.assert_called_once_with("quarterly report", data_sources=["slack"])

    async def test_ainvoke_with_data_sources(self) -> None:
        """Test the full async LangChain tool-call path with data_sources."""
        result = await self.tool.ainvoke({"query": "quarterly report", "data_sources": ["slack"]})

        assert "Q2 Sales Report" in result
        self.mock_retriever.ainvoke.assert_called_once_with("quarterly report", data_sources=["slack"])

    async def test_arun_with_no_results(self) -> None:
        """Test the _arun method when no results are found."""
        self.mock_retriever.ainvoke.return_value = []

        result = await self.tool._arun("nothing matches this")

        assert result == "No results found."

    async def test_arun_with_error(self) -> None:
        """Test the _arun method when an error occurs."""
        self.mock_retriever.ainvoke.side_effect = Exception("Test error")

        result = await self.tool._arun("quarterly report")

        assert "Error running Glean search: Test error" in result
