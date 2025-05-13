from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_glean.retrievers.people import GleanPeopleProfileRetriever, PeopleProfileBasicRequest
from langchain_glean.tools.people_profile_search import GleanPeopleProfileSearchTool


class TestGleanPeopleProfileSearchTool:
    """Test the GleanPeopleProfileSearchTool class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        # Create mock retriever
        self.mock_retriever = MagicMock(spec=GleanPeopleProfileRetriever)

        # Set up sample documents for retriever responses
        self.sample_doc1 = Document(
            page_content="Jane Doe\nSoftware Engineer",
            metadata={
                "email": "jane@example.com",
                "location": "New York",
                "department": "Engineering",
                "title": "Software Engineer",
                "url": "https://example.com/people/jane",
            },
        )

        self.sample_doc2 = Document(
            page_content="John Smith\nProduct Manager",
            metadata={
                "email": "john@example.com",
                "location": "San Francisco",
                "department": "Product",
                "title": "Product Manager",
                "url": "https://example.com/people/john",
            },
        )

        # Set up mock responses
        self.mock_retriever.invoke.return_value = [self.sample_doc1, self.sample_doc2]
        self.mock_retriever.ainvoke.return_value = [self.sample_doc1, self.sample_doc2]

        # Initialize the tool with the mock retriever
        self.tool = GleanPeopleProfileSearchTool(retriever=self.mock_retriever)

        yield

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        assert self.tool.name == "people_profile_search"
        assert self.tool.description == "Search for people within the organization by name, email, or other keywords."
        assert self.tool.retriever == self.mock_retriever
        assert self.tool.args_schema == PeopleProfileBasicRequest
        assert not self.tool.return_direct

    def test_run_with_string_query(self) -> None:
        """Test the _run method with a string query."""
        result = self.tool._run("software engineer")

        # Verify the retriever's invoke method was called with the correct parameters
        self.mock_retriever.invoke.assert_called_once_with("software engineer")

        # Check the output format now matches our new formatting
        expected_result_snippet_1 = "- Jane Doe\n  Title: Software Engineer"
        expected_result_snippet_2 = "Email: jane@example.com"
        expected_result_snippet_3 = "Department: Engineering"

        assert expected_result_snippet_1 in result
        assert expected_result_snippet_2 in result
        assert expected_result_snippet_3 in result

    def test_run_with_typed_request(self) -> None:
        """Test the _run method with a PeopleProfileBasicRequest."""
        request = PeopleProfileBasicRequest(query="engineer", filters={"department": "Engineering"}, page_size=10)

        result = self.tool._run(request)

        # Verify the retriever's invoke method was called with the correct request
        self.mock_retriever.invoke.assert_called_once_with(request)

        # Check the output format now matches our new formatting
        expected_result_snippet_1 = "- Jane Doe\n  Title: Software Engineer"
        expected_result_snippet_2 = "Email: jane@example.com"
        expected_result_snippet_3 = "Department: Engineering"

        assert expected_result_snippet_1 in result
        assert expected_result_snippet_2 in result
        assert expected_result_snippet_3 in result

    def test_run_with_no_results(self) -> None:
        """Test the _run method when no results are found."""
        # Update the mock to return an empty list
        self.mock_retriever.invoke.return_value = []

        result = self.tool._run("nonexistent person")

        assert result.startswith("No matching people found.")

    def test_run_with_error(self) -> None:
        """Test the _run method when an error occurs."""
        # Make the mock raise an exception
        self.mock_retriever.invoke.side_effect = Exception("Test error")

        result = self.tool._run("software engineer")

        assert "Error searching people profiles: Test error" in result

    async def test_arun_with_string_query(self) -> None:
        """Test the _arun method with a string query."""
        result = await self.tool._arun("software engineer")

        # Verify the retriever's ainvoke method was called with the correct parameters
        self.mock_retriever.ainvoke.assert_called_once_with("software engineer")

        # Check the output format now matches our new formatting
        expected_result_snippet_1 = "- Jane Doe\n  Title: Software Engineer"
        expected_result_snippet_2 = "Email: jane@example.com"
        expected_result_snippet_3 = "Department: Engineering"

        assert expected_result_snippet_1 in result
        assert expected_result_snippet_2 in result
        assert expected_result_snippet_3 in result

    async def test_arun_with_typed_request(self) -> None:
        """Test the _arun method with a PeopleProfileBasicRequest."""
        request = PeopleProfileBasicRequest(query="engineer", filters={"department": "Engineering"}, page_size=10)

        result = await self.tool._arun(request)

        # Verify the retriever's ainvoke method was called with the correct request
        self.mock_retriever.ainvoke.assert_called_once_with(request)

        # Check the output format now matches our new formatting
        expected_result_snippet_1 = "- Jane Doe\n  Title: Software Engineer"
        expected_result_snippet_2 = "Email: jane@example.com"
        expected_result_snippet_3 = "Department: Engineering"

        assert expected_result_snippet_1 in result
        assert expected_result_snippet_2 in result
        assert expected_result_snippet_3 in result

    async def test_arun_with_no_results(self) -> None:
        """Test the _arun method when no results are found."""
        # Update the mock to return an empty list
        self.mock_retriever.ainvoke.return_value = []

        result = await self.tool._arun("nonexistent person")

        assert result.startswith("No matching people found.")

    async def test_arun_with_error(self) -> None:
        """Test the _arun method when an error occurs."""
        # Make the mock raise an exception
        self.mock_retriever.ainvoke.side_effect = Exception("Test error")

        result = await self.tool._arun("software engineer")

        assert "Error searching people profiles: Test error" in result
