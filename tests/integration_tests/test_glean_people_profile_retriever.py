import os
import unittest
from typing import List, Type

from dotenv import load_dotenv
from glean.api_client.models import (
    FacetFilter,
    FacetFilterValue,
    ListEntitiesRequest,
    RelationType,
)
from langchain_core.documents import Document

from langchain_glean.retrievers.people import GleanPeopleProfileRetriever, PeopleProfileBasicRequest


class TestGleanPeopleProfileRetriever(unittest.TestCase):
    """Integration tests for the GleanPeopleProfileRetriever."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def retriever_constructor(self) -> Type[GleanPeopleProfileRetriever]:
        """Get the retriever constructor for integration tests."""
        return GleanPeopleProfileRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        """Get the parameters for the retriever constructor."""
        return {}

    @property
    def retriever_query_example(self) -> str:
        """Returns an example query for the retriever."""
        return "engineer"

    def test_invoke_returns_documents(self) -> None:
        """Test that invoke returns documents."""
        retriever = self.retriever_constructor(**self.retriever_constructor_params)
        docs = retriever.invoke(self.retriever_query_example)
        self.assertIsInstance(docs, List)
        if docs:
            self.assertIsInstance(docs[0], Document)
            self.assertTrue(docs[0].page_content)
            self.assertIn("email", docs[0].metadata)
            self.assertIn("title", docs[0].metadata)

    def test_ainvoke_returns_documents(self) -> None:
        """Test that ainvoke returns documents."""
        import asyncio

        async def _test():
            retriever = self.retriever_constructor(**self.retriever_constructor_params)
            docs = await retriever.ainvoke(self.retriever_query_example)
            self.assertIsInstance(docs, List)
            if docs:
                self.assertIsInstance(docs[0], Document)
                self.assertTrue(docs[0].page_content)
                self.assertIn("email", docs[0].metadata)

        asyncio.run(_test())

    def test_invoke_with_k_constructor_param(self) -> None:
        """Test that k constructor param works."""
        retriever = self.retriever_constructor(k=1, **self.retriever_constructor_params)
        docs = retriever.invoke(self.retriever_query_example)
        self.assertIsInstance(docs, List)
        if docs:
            self.assertLessEqual(len(docs), 1)

    def test_invoke_with_basic_request(self) -> None:
        """Test invoke with a PeopleProfileBasicRequest."""
        retriever = self.retriever_constructor(**self.retriever_constructor_params)

        request = PeopleProfileBasicRequest(query="engineer", page_size=2)

        docs = retriever.invoke(request)

        self.assertIsInstance(docs, List)

        if docs:
            self.assertIsInstance(docs[0], Document)
            self.assertLessEqual(len(docs), 2)

    def test_invoke_with_filters(self) -> None:
        """Test invoke with filters."""
        retriever = self.retriever_constructor(**self.retriever_constructor_params)

        request = PeopleProfileBasicRequest(filters={"department": "Engineering"}, page_size=5)

        docs = retriever.invoke(request)

        self.assertIsInstance(docs, List)

        if docs:
            self.assertIsInstance(docs[0], Document)
            for doc in docs:
                self.assertEqual(doc.metadata.get("department"), "Engineering")

    def test_invoke_with_native_request(self) -> None:
        """Test invoke with a native ListEntitiesRequest."""
        retriever = self.retriever_constructor(**self.retriever_constructor_params)

        entities_request = ListEntitiesRequest(
            entity_type="PEOPLE",
            query="manager",
            page_size=3,
            filter=[FacetFilter(field_name="title", values=[FacetFilterValue(value="Manager", relation_type=RelationType.EQUALS)])],
        )

        docs = retriever.invoke(entities_request)

        self.assertIsInstance(docs, List)

        if docs:
            self.assertIsInstance(docs[0], Document)
            for doc in docs:
                self.assertIn("Manager", doc.metadata.get("title", ""), f"Title '{doc.metadata.get('title')}' doesn't contain 'Manager'")

    def test_combined_advanced_query(self) -> None:
        """Test with multiple filters."""
        retriever = self.retriever_constructor(**self.retriever_constructor_params)

        entities_request = ListEntitiesRequest(
            entity_type="PEOPLE",
            page_size=5,
            filter=[
                FacetFilter(field_name="department", values=[FacetFilterValue(value="Engineering", relation_type=RelationType.EQUALS)]),
                FacetFilter(field_name="title", values=[FacetFilterValue(value="Senior", relation_type=RelationType.EQUALS)]),
            ],
        )

        docs = retriever.invoke(entities_request)

        self.assertIsInstance(docs, List)

        if docs:
            for doc in docs:
                self.assertEqual(doc.metadata.get("department"), "Engineering")
                self.assertIn("Senior", doc.metadata.get("title", ""), f"Title '{doc.metadata.get('title')}' doesn't contain 'Senior'")
