from typing import Type

from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
    ToolsIntegrationTests,
)

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool


class TestGleanRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[GleanSearchRetriever]:
        """Get the retriever constructor for integration tests."""
        return GleanSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        """Get the parameters for the retriever constructor."""
        return {
            "subdomain": "test-glean",
            "api_token": "test-token",
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns an example query for the retriever.
        """
        return "example query"


class TestGleanSearchRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[GleanSearchRetriever]:
        """Get the retriever constructor for integration tests."""
        return GleanSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        """Get the parameters for the retriever constructor."""
        return {
            "subdomain": "test-glean",
            "api_token": "test-token",
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns an example query for the retriever.
        """
        return "search query example"


class TestGleanSearchTool(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[GleanSearchTool]:
        """Get the tool constructor for integration tests."""
        return GleanSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        """Get the parameters for the tool constructor."""
        # Create a retriever for the tool
        retriever = GleanSearchRetriever(
            subdomain="test-glean",
            api_token="test-token",
        )

        return {"retriever": retriever}

    @property
    def tool_input_example(self) -> dict:
        """
        Returns an example input for the tool.
        """
        return {
            "query": "example query",
            "page_size": 100,
            "disable_spellcheck": True,
            "request_options": {"facetFilters": [{"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}]}]},
        }
