import os
import unittest
from typing import Type

from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)

from langchain_glean.retrievers import GleanSearchRetriever


class TestGleanSearchRetriever(unittest.TestCase, RetrieversIntegrationTests):
    def setUp(self) -> None:
        """Set up test environment variables."""
        unittest.TestCase.setUp(self)
        os.environ["GLEAN_SUBDOMAIN"] = "test-glean"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "test@example.com"

    def tearDown(self) -> None:
        """Clean up test environment variables."""
        unittest.TestCase.tearDown(self)
        for var in ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "GLEAN_ACT_AS"]:
            os.environ.pop(var, None)

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
        return "search query example"
