import os
from types import SimpleNamespace
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
        os.environ["GLEAN_INSTANCE"] = "test-glean"
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

        # Create mock sample data with SimpleNamespace for better attribute access
        self.mock_author = SimpleNamespace(name="John Doe", email="john@example.com")

        self.mock_doc_metadata = SimpleNamespace(
            datasourceInstance="workspace",
            objectType="Message",
            mimeType="text/plain",
            documentId="doc-123",
            loggingId="log-123",
            createTime="2023-01-01T00:00:00Z",
            updateTime="2023-01-02T00:00:00Z",
            visibility="PUBLIC_VISIBLE",
            documentCategory="PUBLISHED_CONTENT",
            author=self.mock_author,
        )

        self.mock_document = SimpleNamespace(
            id="doc-123",
            datasource="slack",
            doc_type="Message",
            title="Sample Document",
            url="https://example.com/doc",
            metadata=self.mock_doc_metadata,
        )

        self.mock_snippet1 = SimpleNamespace(text="This is a sample snippet.", ranges=[SimpleNamespace(startIndex=0, endIndex=4, type="BOLD")])

        self.mock_snippet2 = SimpleNamespace(text="This is another sample snippet.", ranges=[])

        self.mock_result = SimpleNamespace(
            tracking_token="sample-token",
            document=self.mock_document,
            title="Sample Document",
            url="https://example.com/doc",
            snippets=[self.mock_snippet1, self.mock_snippet2],
        )

        # Create mock search response with our SimpleNamespace objects
        mock_results = MagicMock()
        mock_results.results = [self.mock_result]

        # Mock the query and query_async methods
        mock_search.query.return_value = mock_results
        mock_search.query_async.return_value = mock_results

        self.retriever = GleanSearchRetriever()
        self.retriever._client = mock_client

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    def test_init(self) -> None:
        """Test the initialization of the retriever."""
        assert self.retriever.instance == "test-glean"
        assert self.retriever.api_token == "test-token"
        assert self.retriever.act_as == "test@example.com"

        self.mock_glean.assert_called_once_with(
            api_token="test-token",
            instance="test-glean",
        )

    def test_init_with_missing_env_vars(self) -> None:
        """Test initialization with missing environment variables."""
        del os.environ["GLEAN_INSTANCE"]
        del os.environ["GLEAN_API_TOKEN"]

        with pytest.raises(ValueError):
            GleanSearchRetriever()

    def test_invoke(self) -> None:
        """Test the invoke method."""
        docs = self.retriever.invoke("test query")

        # Verify the search.query method was called with the correct parameters
        self.retriever._client.search.query.assert_called_once()
        call_args = self.retriever._client.search.query.call_args
        search_request = call_args[1]["request"]

        # Check the SearchRequest object properties
        assert search_request.query == "test query"

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
        # Mock the _build_search_request method to avoid conversion issues in tests
        with patch.object(self.retriever, "_build_search_request") as mock_build:
            search_request_mock = MagicMock()
            mock_build.return_value = search_request_mock

            self.retriever.invoke(
                "test query",
                page_size=20,
                disable_spellcheck=True,
                max_snippet_size=100,
                request_options={
                    "facet_filters": [
                        {"field_name": "datasource", "values": [{"value": "slack", "relation_type": "EQUALS"}, {"value": "gdrive", "relation_type": "EQUALS"}]}
                    ]
                },
            )

            # Verify _build_search_request was called with the correct parameters
            mock_build.assert_called_once()

            # The first positional argument should be the query
            args, kwargs = mock_build.call_args
            assert len(args) > 0
            assert args[0] == "test query"

            # Check that the keyword arguments are correct
            assert "page_size" in kwargs
            assert kwargs["page_size"] == 20
            assert "disable_spellcheck" in kwargs
            assert kwargs["disable_spellcheck"] is True
            assert "max_snippet_size" in kwargs
            assert kwargs["max_snippet_size"] == 100

            # Verify that query was called with the mocked search request
            self.retriever._client.search.query.assert_called_once_with(request=search_request_mock)

    def test_build_document(self) -> None:
        """Test the _build_document method."""
        result = self.retriever._client.search.query.return_value.results[0]

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
