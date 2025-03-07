from typing import Type

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestGleanRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[GleanSearchRetriever]:
        """Get an empty vectorstore for unit tests."""
        return GleanSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a dictionary representing the "args" of an example retriever call.
        """
        return "example query"
