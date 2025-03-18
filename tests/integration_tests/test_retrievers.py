import os
import unittest
from typing import List, Type

from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever


class TestGleanSearchRetriever(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_SUBDOMAIN") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    @property
    def retriever_constructor(self) -> Type[GleanSearchRetriever]:
        """Get the retriever constructor for integration tests."""
        return GleanSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        """Get the parameters for the retriever constructor."""
        return {}  # No params needed as we use environment variables

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
        # First check if the retriever accepts k as a parameter
        try:
            # Use k parameter for GleanSearchRetriever
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
