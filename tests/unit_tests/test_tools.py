from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool


class TestGleanSearchTool:
    """Test the GleanSearchTool class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test."""
        self.mock_retriever = MagicMock(spec=GleanSearchRetriever)

        self.sample_doc = Document(
            page_content="This is a sample document.",
            metadata={
                "title": "Sample Document",
                "url": "https://example.com/doc",
                "document_id": "doc-123",
                "datasource": "slack",
                "doc_type": "Message",
                "author": "John Doe",
                "create_time": "2023-01-01T00:00:00Z",
                "update_time": "2023-01-02T00:00:00Z",
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
        assert not self.tool.return_direct

    def test_run_with_string(self) -> None:
        """Test the _run method with a string query."""
        result = self.tool._run("test query")

        self.mock_retriever.invoke.assert_called_once_with("test query")

        assert "Result 1:" in result
        assert "Title: Sample Document" in result
        assert "URL: https://example.com/doc" in result
        assert "Content: This is a sample document." in result

    def test_run_with_dict(self) -> None:
        """Test the _run method with a dictionary query."""

        result = self.tool._run(
            {
                "query": "test query",
                "page_size": 20,
                "disable_spellcheck": True,
                "request_options": {"facetFilters": [{"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}]}]},
            }
        )

        self.mock_retriever.invoke.assert_called_once_with(
            "test query",
            page_size=20,
            disable_spellcheck=True,
            request_options={"facetFilters": [{"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}]}]},
        )

        assert "Result 1:" in result
        assert "Title: Sample Document" in result
        assert "URL: https://example.com/doc" in result
        assert "Content: This is a sample document." in result

    def test_run_with_no_results(self) -> None:
        """Test the _run method when no results are found."""
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("test query")

        assert result == "No results found."

    def test_run_with_error(self) -> None:
        """Test the _run method when an error occurs."""
        self.mock_retriever.invoke.side_effect = Exception("Test error")

        result = self.tool._run("test query")

        assert result == "Error searching Glean: Test error"

    async def test_arun(self) -> None:
        """Test the _arun method."""
        result = await self.tool._arun("test query")

        self.mock_retriever.ainvoke.assert_called_once_with("test query")

        assert "Result 1:" in result
        assert "Title: Sample Document" in result
        assert "URL: https://example.com/doc" in result
        assert "Content: This is a sample document." in result
