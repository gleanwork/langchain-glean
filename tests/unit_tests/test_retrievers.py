"""Unit tests for Glean retrievers."""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever


class TestGleanSearchRetriever(unittest.TestCase):
    """Test the GleanSearchRetriever class."""

    def setUp(self):
        """Set up the test."""

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

        self.mock_client.post.return_value = self.sample_result

        self.auth_patcher = patch("langchain_glean.retrievers.search.GleanAuth")
        self.mock_auth_class = self.auth_patcher.start()
        self.mock_auth_class.return_value = self.mock_auth

        self.client_patcher = patch("langchain_glean.retrievers.search.GleanClient")
        self.mock_client_class = self.client_patcher.start()
        self.mock_client_class.return_value = self.mock_client

        self.retriever = GleanSearchRetriever(subdomain="test-glean", api_token="test-token", act_as="test@example.com")

    def tearDown(self):
        """Tear down the test."""
        self.auth_patcher.stop()
        self.client_patcher.stop()

    def test_init(self):
        """Test the initialization of the retriever."""
        self.assertEqual(self.retriever.subdomain, "test-glean")
        self.assertEqual(self.retriever.api_token, "test-token")
        self.assertEqual(self.retriever.act_as, "test@example.com")

        self.mock_auth_class.assert_called_once_with(api_token="test-token", subdomain="test-glean", act_as="test@example.com")
        self.mock_client_class.assert_called_once_with(auth=self.mock_auth)

    def test_get_relevant_documents(self):
        """Test the get_relevant_documents method."""
        docs = self.retriever.get_relevant_documents("test query")

        self.mock_client.post.assert_called_once()
        call_args = self.mock_client.post.call_args
        self.assertEqual(call_args[0][0], "search")

        payload = json.loads(call_args[1]["data"])
        self.assertEqual(payload["query"], "test query")
        self.assertEqual(payload["pageSize"], 100)

        self.assertEqual(call_args[1]["headers"], {"Content-Type": "application/json"})

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.page_content, "This is a sample snippet.\nThis is another sample snippet.")

        self.assertEqual(doc.metadata["title"], "Sample Document")
        self.assertEqual(doc.metadata["url"], "https://example.com/doc")
        self.assertEqual(doc.metadata["document_id"], "doc-123")
        self.assertEqual(doc.metadata["datasource"], "slack")
        self.assertEqual(doc.metadata["doc_type"], "Message")
        self.assertEqual(doc.metadata["author"], "John Doe")
        self.assertEqual(doc.metadata["create_time"], "2023-01-01T00:00:00Z")
        self.assertEqual(doc.metadata["update_time"], "2023-01-02T00:00:00Z")

    def test_get_relevant_documents_with_params(self):
        """Test the get_relevant_documents method with additional parameters."""
        self.retriever.get_relevant_documents(
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
        self.assertEqual(payload["query"], "test query")
        self.assertEqual(payload["pageSize"], 20)
        self.assertEqual(payload["disableSpellcheck"], True)
        self.assertEqual(payload["maxSnippetSize"], 100)

        facet_filters = payload["requestOptions"]["facetFilters"]
        self.assertEqual(len(facet_filters), 1)
        self.assertEqual(facet_filters[0]["fieldName"], "datasource")
        self.assertEqual(len(facet_filters[0]["values"]), 2)
        self.assertEqual(facet_filters[0]["values"][0]["value"], "slack")
        self.assertEqual(facet_filters[0]["values"][0]["relationType"], "EQUALS")
        self.assertEqual(facet_filters[0]["values"][1]["value"], "gdrive")
        self.assertEqual(facet_filters[0]["values"][1]["relationType"], "EQUALS")

    def test_build_document(self):
        """Test the _build_document method."""
        result = self.sample_result["results"][0]

        doc = self.retriever._build_document(result)

        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.page_content, "This is a sample snippet.\nThis is another sample snippet.")

        self.assertEqual(doc.metadata["title"], "Sample Document")
        self.assertEqual(doc.metadata["url"], "https://example.com/doc")
        self.assertEqual(doc.metadata["document_id"], "doc-123")
        self.assertEqual(doc.metadata["datasource"], "slack")
        self.assertEqual(doc.metadata["doc_type"], "Message")
        self.assertEqual(doc.metadata["author"], "John Doe")
        self.assertEqual(doc.metadata["create_time"], "2023-01-01T00:00:00Z")
        self.assertEqual(doc.metadata["update_time"], "2023-01-02T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
