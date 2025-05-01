import json
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever


class TestGleanSearchRetriever:
    """Test the GleanSearchRetriever class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test."""
        # Set environment variables for testing
        os.environ["GLEAN_SUBDOMAIN"] = "test-glean"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "test@example.com"

        # Mock the Glean class
        self.mock_glean_patcher = patch("langchain_glean.retrievers.search.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Mock the client property of the Glean instance
        mock_client = MagicMock()
        self.mock_glean.return_value.client = mock_client

        # Create mock search client
        mock_search = MagicMock()
        mock_client.search = mock_search

        # Sample search response with a document
        search_result = MagicMock()
        search_result.results = [
            {
                "trackingToken": "sample-token",
                "document": {
                    "id": "doc-123",
                    "datasource": "slack",
                    "docType": "Message",
                    "title": "Sample Document",
                    "url": "https://example.com/doc",
                    "metadata": {
                        "datasource": "slack",
                        "datasourceInstance": "workspace",
                        "objectType": "Message",
                        "mimeType": "text/plain",
                        "documentId": "doc-123",
                        "loggingId": "log-123",
                        "createTime": "2023-01-01T00:00:00Z",
                        "updateTime": "2023-01-02T00:00:00Z",
                        "visibility": "PUBLIC_VISIBLE",
                        "documentCategory": "PUBLISHED_CONTENT",
                        "author": {"name": "John Doe", "email": "john@example.com"},
                    },
                },
                "title": "Sample Document",
                "url": "https://example.com/doc",
                "snippets": [
                    {"text": "This is a sample snippet.", "ranges": [{"startIndex": 0, "endIndex": 4, "type": "BOLD"}]},
                    {"text": "This is another sample snippet.", "ranges": []},
                ],
            }
        ]

        # Mock the execute and execute_async methods
        mock_search.execute.return_value = search_result
        mock_search.execute_async.return_value = search_result

        self.retriever = GleanSearchRetriever()
        self.retriever._client = mock_client

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_init(self) -> None:
        """Test the initialization of the retriever."""
        assert self.retriever.subdomain == "test-glean"
        assert self.retriever.api_token == "test-token"
        assert self.retriever.act_as == "test@example.com"

        self.mock_glean.assert_called_once_with(
            bearer_auth="test-token",
            domain="test-glean",
        )

    def test_init_with_missing_env_vars(self) -> None:
        """Test initialization with missing environment variables."""
        del os.environ["GLEAN_SUBDOMAIN"]
        del os.environ["GLEAN_API_TOKEN"]

        with pytest.raises(ValueError):
            GleanSearchRetriever()

    def test_invoke(self) -> None:
        """Test the invoke method."""
        docs = self.retriever.invoke("test query")

        # Verify the search.execute method was called with the correct parameters
        self.retriever._client.search.execute.assert_called_once()
        call_args = self.retriever._client.search.execute.call_args
        search_request = call_args[1]["search_request"]
        assert "query" in search_request
        assert search_request["query"] == "test query"

        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, Document)
        assert doc.page_content == "This is a sample snippet.\nThis is another sample snippet."

        assert doc.metadata["title"] == "Sample Document"
        assert doc.metadata["url"] == "https://example.com/doc"
        assert doc.metadata["document_id"] == "doc-123"
        assert doc.metadata["datasource"] == "slack"
        assert doc.metadata["doc_type"] == "Message"
        assert doc.metadata["author"] == "John Doe"
        assert doc.metadata["create_time"] == "2023-01-01T00:00:00Z"
        assert doc.metadata["update_time"] == "2023-01-02T00:00:00Z"

    def test_invoke_with_params(self) -> None:
        """Test the invoke method with additional parameters."""
        self.retriever.invoke(
            "test query",
            page_size=20,
            disable_spellcheck=True,
            max_snippet_size=100,
            request_options={
                "facetFilters": [
                    {"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}, {"value": "gdrive", "relationType": "EQUALS"}]}
                ]
            },
        )

        call_args = self.retriever._client.search.execute.call_args
        search_request = call_args[1]["search_request"]
        assert search_request["query"] == "test query"
        assert search_request["page_size"] == 20
        assert search_request["disable_spellcheck"] is True
        assert search_request["max_snippet_size"] == 100

        facet_filters = search_request["request_options"]["facetFilters"]
        assert len(facet_filters) == 1
        assert facet_filters[0]["fieldName"] == "datasource"
        assert len(facet_filters[0]["values"]) == 2
        assert facet_filters[0]["values"][0]["value"] == "slack"
        assert facet_filters[0]["values"][0]["relationType"] == "EQUALS"
        assert facet_filters[0]["values"][1]["value"] == "gdrive"
        assert facet_filters[0]["values"][1]["relationType"] == "EQUALS"

    def test_build_document(self) -> None:
        """Test the _build_document method."""
        result = self.retriever._client.search.execute.return_value.results[0]

        doc = self.retriever._build_document(result)

        assert isinstance(doc, Document)
        assert doc.page_content == "This is a sample snippet.\nThis is another sample snippet."

        assert doc.metadata["title"] == "Sample Document"
        assert doc.metadata["url"] == "https://example.com/doc"
        assert doc.metadata["document_id"] == "doc-123"
        assert doc.metadata["datasource"] == "slack"
        assert doc.metadata["doc_type"] == "Message"
        assert doc.metadata["author"] == "John Doe"
        assert doc.metadata["create_time"] == "2023-01-01T00:00:00Z"
        assert doc.metadata["update_time"] == "2023-01-02T00:00:00Z"
