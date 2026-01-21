import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from glean.api_client.models import (
    FacetFilter,
    FacetFilterValue,
    RelationType,
    SearchRequest,
    SearchRequestOptions,
)
from langchain_core.documents import Document

from langchain_glean.retrievers.search import GleanSearchRetriever


class TestGleanSearchRetriever:
    """Test the GleanSearchRetriever class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test."""
        # Set environment variables for testing
        os.environ["GLEAN_INSTANCE"] = "test-glean"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "test@example.com"

        # Mock the Glean class where it's directly used
        self.mock_glean_patcher = patch("langchain_glean.retrievers.search.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Create mock search client
        mock_search = MagicMock()
        mock_client = MagicMock()
        self.mock_glean.return_value.__enter__.return_value.client = mock_client
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

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

    # ===== BASIC TESTS =====

    def test_init(self) -> None:
        """Test the initialization of the retriever."""
        assert self.retriever.instance == "test-glean"
        assert self.retriever.api_token == "test-token"
        assert self.retriever.act_as == "test@example.com"
        assert self.retriever.k == 10

    def test_init_with_missing_env_vars(self) -> None:
        """Test initialization with missing environment variables."""
        del os.environ["GLEAN_INSTANCE"]
        del os.environ["GLEAN_API_TOKEN"]

        with pytest.raises(ValueError):
            GleanSearchRetriever()

    def test_invoke_with_simple_query(self) -> None:
        """Test the invoke method with a simple string query."""
        docs = self.retriever.invoke("test query")

        # Verify the search.query method was called with the correct parameters
        self.mock_glean.return_value.__enter__.return_value.client.search.query.assert_called_once()
        call_args = self.mock_glean.return_value.__enter__.return_value.client.search.query.call_args

        # Check the unpacked kwargs (SDK 0.11+ uses individual params, not request=)
        assert call_args[1]["query"] == "test query"

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

    def test_invoke_with_basic_params(self) -> None:
        """Test the invoke method with basic additional parameters."""
        # Mock the _build_search_request method to avoid conversion issues in tests
        with patch.object(self.retriever, "_build_search_request") as mock_build:
            search_request_mock = MagicMock()
            mock_build.return_value = search_request_mock

            self.retriever.invoke("test query", page_size=20, disable_spellcheck=True, max_snippet_size=100)

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

            # Verify that query was called (SDK 0.11+ unpacks request into kwargs)
            self.mock_glean.return_value.__enter__.return_value.client.search.query.assert_called_once()

    def test_build_document(self) -> None:
        """Test the _build_document method."""
        result = self.mock_glean.return_value.__enter__.return_value.client.search.query.return_value.results[0]

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

    # ===== ADVANCED TESTS =====

    def test_invoke_with_native_search_request(self):
        """Test invoking with a native SearchRequest object."""
        # Create a strongly typed SearchRequest
        search_request = SearchRequest(
            query="test query",
            page_size=15,
            disable_spellcheck=True,
            max_snippet_size=150,
            request_options=SearchRequestOptions(
                fetch_all_datasource_counts=True, response_hints=["RESULTS", "FACET_RESULTS"], datasources_filter=["slack", "gmail"], facet_bucket_size=20
            ),
        )

        docs = self.retriever.invoke(search_request)

        # Verify the search was called with unpacked request params (SDK 0.11+)
        self.mock_glean.return_value.__enter__.return_value.client.search.query.assert_called_once()
        call_args = self.mock_glean.return_value.__enter__.return_value.client.search.query.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["http_headers"] == {"X-Glean-ActAs": "test@example.com"}

        # Verify we got documents back
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "This is a sample snippet.\nThis is another sample snippet."

    def test_invoke_with_partial_native_options(self):
        """Test invoking with a partial native SearchRequestOptions object."""
        # Create just the options part
        request_options = SearchRequestOptions(datasources_filter=["confluence", "drive"], fetch_all_datasource_counts=True, facet_bucket_size=30)

        _ = self.retriever.invoke("test query", page_size=20, request_options=request_options)

        # Verify the search call
        self.mock_glean.return_value.__enter__.return_value.client.search.query.assert_called_once()

    def test_invoke_with_facet_filters(self):
        """Test invoking with strongly typed facet filters."""
        facet_filters = [
            FacetFilter(
                field_name="datasource",
                values=[FacetFilterValue(value="slack", relation_type=RelationType.EQUALS), FacetFilterValue(value="drive", relation_type=RelationType.EQUALS)],
            ),
            FacetFilter(field_name="time", values=[FacetFilterValue(value="2023-01-01", relation_type=RelationType.GT)]),
        ]

        # Create a SearchRequestOptions with facet_filters
        request_options = SearchRequestOptions(facet_filters=facet_filters, facet_bucket_size=20)

        _ = self.retriever.invoke("test query", request_options=request_options)

        # Verify the search call
        self.mock_glean.return_value.__enter__.return_value.client.search.query.assert_called_once()

    async def test_ainvoke_with_native_search_request(self):
        """Test async invoking with a native SearchRequest object."""
        # Create a strongly typed SearchRequest
        search_request = SearchRequest(
            query="test query",
            page_size=15,
            disable_spellcheck=True,
            max_snippet_size=150,
            request_options=SearchRequestOptions(
                fetch_all_datasource_counts=True, response_hints=["RESULTS", "FACET_RESULTS"], datasources_filter=["slack", "gmail"], facet_bucket_size=20
            ),
        )

        # Set up mock response for async call that has the same structure as the sync responses
        mock_async_results = SimpleNamespace(results=[self.mock_result])

        # Replace the original MagicMock with a proper coroutine function
        async def mock_query_async(*args, **kwargs):
            return mock_async_results

        self.mock_glean.return_value.__enter__.return_value.client.search.query_async = mock_query_async

        docs = await self.retriever.ainvoke(search_request)

        # Verify we got documents back
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "This is a sample snippet.\nThis is another sample snippet."

    def test_combining_with_limit_parameter(self):
        """Test combining k parameter with SearchRequest."""
        # Create a SearchRequest
        search_request = SearchRequest(query="test query", request_options=SearchRequestOptions(datasources_filter=["confluence"], facet_bucket_size=20))

        # Call with k parameter
        _ = self.retriever.invoke(search_request, k=5)

        # Verify the search call
        self.mock_glean.return_value.__enter__.return_value.client.search.query.assert_called_once()
