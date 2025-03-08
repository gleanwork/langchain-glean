import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool


class TestGleanSearchTool(unittest.TestCase):
    """Test the GleanSearchTool class."""

    def setUp(self) -> None:
        """Set up the test."""
        # Create a mock retriever
        self.mock_retriever = MagicMock(spec=GleanSearchRetriever)

        # Create a sample document
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

        # Set up the mock retriever to return the sample document
        self.mock_retriever.get_relevant_documents.return_value = [self.sample_doc]
        self.mock_retriever.aget_relevant_documents.return_value = [self.sample_doc]

        # Create the tool
        self.tool = GleanSearchTool(retriever=self.mock_retriever)

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        self.assertEqual(self.tool.name, "glean_search")
        self.assertEqual(self.tool.retriever, self.mock_retriever)
        self.assertFalse(self.tool.return_direct)

    def test_run_with_string(self) -> None:
        """Test the _run method with a string query."""
        # Call the method
        result = self.tool._run("test query")

        # Check that the retriever was called correctly
        self.mock_retriever.get_relevant_documents.assert_called_once_with("test query")

        # Check the result
        self.assertIn("Result 1:", result)
        self.assertIn("Title: Sample Document", result)
        self.assertIn("URL: https://example.com/doc", result)
        self.assertIn("Content: This is a sample document.", result)

    def test_run_with_dict(self) -> None:
        """Test the _run method with a dictionary query."""
        # Call the method
        result = self.tool._run(
            {
                "query": "test query",
                "page_size": 20,
                "disable_spellcheck": True,
                "request_options": {"facetFilters": [{"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}]}]},
            }
        )

        # Check that the retriever was called correctly
        self.mock_retriever.get_relevant_documents.assert_called_once_with(
            "test query",
            page_size=20,
            disable_spellcheck=True,
            request_options={"facetFilters": [{"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}]}]},
        )

        # Check the result
        self.assertIn("Result 1:", result)
        self.assertIn("Title: Sample Document", result)
        self.assertIn("URL: https://example.com/doc", result)
        self.assertIn("Content: This is a sample document.", result)

    def test_run_with_no_results(self) -> None:
        """Test the _run method when no results are found."""
        # Set up the mock retriever to return no documents
        self.mock_retriever.get_relevant_documents.return_value = []

        # Call the method
        result = self.tool._run("test query")

        # Check the result
        self.assertEqual(result, "No results found.")

    def test_run_with_error(self) -> None:
        """Test the _run method when an error occurs."""
        # Set up the mock retriever to raise an exception
        self.mock_retriever.get_relevant_documents.side_effect = Exception("Test error")

        # Call the method
        result = self.tool._run("test query")

        # Check the result
        self.assertEqual(result, "Error searching Glean: Test error")

    async def test_arun(self) -> None:
        """Test the _arun method."""
        # Call the method
        result = await self.tool._arun("test query")

        # Check that the retriever was called correctly
        self.mock_retriever.aget_relevant_documents.assert_called_once_with("test query")

        # Check the result
        self.assertIn("Result 1:", result)
        self.assertIn("Title: Sample Document", result)
        self.assertIn("URL: https://example.com/doc", result)
        self.assertIn("Content: This is a sample document.", result)


if __name__ == "__main__":
    unittest.main()
