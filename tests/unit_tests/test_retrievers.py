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

        self.mock_client = MagicMock()
        self.mock_auth = MagicMock()

        self.sample_result = {
            "results": [
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
        }

        mock_response = MagicMock()
        self.mock_client.post.return_value = mock_response
        self.mock_client.parse_response.return_value = self.sample_result

        self.auth_patcher = patch("langchain_glean.retrievers.search.GleanAuth")
        self.mock_auth_class = self.auth_patcher.start()
        self.mock_auth_class.return_value = self.mock_auth

        self.client_patcher = patch("langchain_glean.retrievers.search.GleanClient")
        self.mock_client_class = self.client_patcher.start()
        self.mock_client_class.return_value = self.mock_client

        self.retriever = GleanSearchRetriever()

        yield

        # Clean up after tests
        self.auth_patcher.stop()
        self.client_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_init(self) -> None:
        """Test the initialization of the retriever."""
        assert self.retriever.subdomain == "test-glean"
        assert self.retriever.api_token == "test-token"
        assert self.retriever.act_as == "test@example.com"

        self.mock_auth_class.assert_called_once_with(api_token="test-token", subdomain="test-glean", act_as="test@example.com")
        self.mock_client_class.assert_called_once_with(auth=self.mock_auth)

    def test_init_with_missing_env_vars(self) -> None:
        """Test initialization with missing environment variables."""
        del os.environ["GLEAN_SUBDOMAIN"]
        del os.environ["GLEAN_API_TOKEN"]

        with pytest.raises(ValueError):
            GleanSearchRetriever()

    def test_invoke(self) -> None:
        """Test the invoke method."""
        docs = self.retriever.invoke("test query")

        self.mock_client.post.assert_called_once()
        call_args = self.mock_client.post.call_args
        assert call_args[0][0] == "search"

        payload = json.loads(call_args[1]["data"])
        assert payload["query"] == "test query"
        assert payload["pageSize"] == 100

        assert call_args[1]["headers"] == {"Content-Type": "application/json"}

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

        payload = json.loads(self.mock_client.post.call_args[1]["data"])
        assert payload["query"] == "test query"
        assert payload["pageSize"] == 20
        assert payload["disableSpellcheck"] is True
        assert payload["maxSnippetSize"] == 100

        facet_filters = payload["requestOptions"]["facetFilters"]
        assert len(facet_filters) == 1
        assert facet_filters[0]["fieldName"] == "datasource"
        assert len(facet_filters[0]["values"]) == 2
        assert facet_filters[0]["values"][0]["value"] == "slack"
        assert facet_filters[0]["values"][0]["relationType"] == "EQUALS"
        assert facet_filters[0]["values"][1]["value"] == "gdrive"
        assert facet_filters[0]["values"][1]["relationType"] == "EQUALS"

    def test_build_document(self) -> None:
        """Test the _build_document method."""
        result = self.sample_result["results"][0]

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
