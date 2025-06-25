import os
import unittest
from typing import List, Type

from dotenv import load_dotenv
from glean.api_client.models import (
    FacetFilter,
    FacetFilterValue,
    RelationType,
    SearchRequest,
    SearchRequestOptions,
)
from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever


class TestGleanSearchRetriever(unittest.TestCase):
    """Integration tests for the GleanSearchRetriever."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def retriever_constructor(self) -> Type[GleanSearchRetriever]:
        """Get the retriever constructor for integration tests."""
        return GleanSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        """Get the parameters for the retriever constructor."""
        return {}

    @property
    def retriever_query_example(self) -> str:
        """Returns an example query for the retriever."""
        return "What can Glean's assistant do?"

    def test_invoke_returns_documents(self) -> None:
        """Test that invoke returns documents."""
        retriever = self.retriever_constructor(**self.retriever_constructor_params)
        docs = retriever.invoke(self.retriever_query_example)
        self.assertIsInstance(docs, List)
        if docs:
            self.assertIsInstance(docs[0], Document)

    def test_ainvoke_returns_documents(self) -> None:
        """Test that ainvoke returns documents."""
        import asyncio

        async def _test():
            retriever = self.retriever_constructor(**self.retriever_constructor_params)
            docs = await retriever.ainvoke(self.retriever_query_example)
            self.assertIsInstance(docs, List)
            if docs:
                self.assertIsInstance(docs[0], Document)

        asyncio.run(_test())

    def test_invoke_with_k_kwarg(self) -> None:
        """Test that invoke with k kwarg works."""
        retriever = self.retriever_constructor(**self.retriever_constructor_params)
        docs = retriever.invoke(self.retriever_query_example, k=1)
        self.assertIsInstance(docs, List)
        if docs:
            self.assertLessEqual(len(docs), 1)

    def test_k_constructor_param(self) -> None:
        """Test that k constructor param works."""
        try:
            retriever = self.retriever_constructor(k=1, **self.retriever_constructor_params)
            docs = retriever.invoke(self.retriever_query_example)
            self.assertIsInstance(docs, List)
            if docs:
                self.assertLessEqual(len(docs), 1)
        except TypeError as e:
            if "got an unexpected keyword argument 'k'" in str(e):
                self.skipTest("Retriever does not accept k as a constructor parameter")
            else:
                raise

    def test_native_search_request(self) -> None:
        """Test with a fully configured native SearchRequest object."""
        search_request = SearchRequest(
            query="search api",
            page_size=5,
            disable_spellcheck=True,
            max_snippet_size=150,
            request_options=SearchRequestOptions(response_hints=["RESULTS", "FACET_RESULTS", "SPELLCHECK_METADATA"], facet_bucket_size=30),
        )

        retriever = GleanSearchRetriever()
        docs = retriever.invoke(search_request)

        self.assertIsInstance(docs, List)

        if docs:
            self.assertIsInstance(docs[0], Document)
            self.assertTrue(docs[0].page_content)
            self.assertIn("title", docs[0].metadata)
            self.assertIn("url", docs[0].metadata)
            self.assertIn("document_id", docs[0].metadata)
            self.assertIn("datasource", docs[0].metadata)

    def test_invoke_with_facet_filters(self) -> None:
        """Test with strongly typed facet filters."""
        facet_filters = [FacetFilter(field_name="datasource", values=[FacetFilterValue(value="confluence", relation_type=RelationType.EQUALS)])]

        request_options = SearchRequestOptions(facet_filters=facet_filters, facet_bucket_size=10)

        retriever = GleanSearchRetriever()
        docs = retriever.invoke("documentation", request_options=request_options)

        self.assertIsInstance(docs, List)

        if docs:
            self.assertIsInstance(docs[0], Document)
            for doc in docs:
                self.assertEqual(doc.metadata.get("datasource"), "confluence")

    def test_combined_filters_and_parameters(self) -> None:
        """Test combining multiple filter types and parameters."""
        datasource_filter = FacetFilter(field_name="datasource", values=[FacetFilterValue(value="slack", relation_type=RelationType.EQUALS)])

        date_filter = FacetFilter(field_name="create_time", values=[FacetFilterValue(value="2023-01-01", relation_type=RelationType.GT)])

        request_options = SearchRequestOptions(facet_filters=[datasource_filter, date_filter], facet_bucket_size=20, fetch_all_datasource_counts=True)

        retriever = GleanSearchRetriever()
        docs = retriever.invoke("message", page_size=10, request_options=request_options)

        self.assertIsInstance(docs, List)

        if docs:
            for doc in docs:
                self.assertEqual(doc.metadata.get("datasource"), "slack")
                self.assertGreater(doc.metadata.get("create_time", "2023-01-01"), "2023-01-01")

    async def test_async_native_search_request(self) -> None:
        """Test async invoke with a native SearchRequest."""
        import asyncio

        async def _test():
            search_request = SearchRequest(
                query="search api", page_size=5, disable_spellcheck=True, request_options=SearchRequestOptions(response_hints=["RESULTS"])
            )

            retriever = GleanSearchRetriever()
            docs = await retriever.ainvoke(search_request)

            self.assertIsInstance(docs, List)

            if docs:
                self.assertIsInstance(docs[0], Document)
                self.assertTrue(docs[0].page_content)
                self.assertIn("title", docs[0].metadata)

        await asyncio.get_event_loop().create_task(_test())
