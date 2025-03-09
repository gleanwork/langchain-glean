import os
import unittest
from typing import Type

from langchain_tests.integration_tests import (
    ToolsIntegrationTests,
)

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.tools import GleanSearchTool


class TestGleanSearchTool(unittest.TestCase, ToolsIntegrationTests):
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
    def tool_constructor(self) -> Type[GleanSearchTool]:
        """Get the tool constructor for integration tests."""
        return GleanSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        """Get the parameters for the tool constructor."""
        # Create a retriever for the tool
        retriever = GleanSearchRetriever()  # No params needed as we use environment variables
        return {"retriever": retriever}

    @property
    def tool_input_example(self) -> dict:
        """Returns an example input for the tool."""
        return {
            "query": "example query",
            "page_size": 100,
            "disable_spellcheck": True,
            "request_options": {"facetFilters": [{"fieldName": "datasource", "values": [{"value": "slack", "relationType": "EQUALS"}]}]},
        }
