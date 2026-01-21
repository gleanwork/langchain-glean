import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from glean.api_client.models import (
    FacetFilter,
    FacetFilterValue,
    ListEntitiesRequest,
    RelationType,
)
from langchain_core.documents import Document

from langchain_glean.retrievers.people import GleanPeopleProfileRetriever, PeopleProfileBasicRequest


class TestGleanPeopleProfileRetriever:
    """Test the GleanPeopleProfileRetriever class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        # Set environment variables for testing
        os.environ["GLEAN_INSTANCE"] = "test-glean"
        os.environ["GLEAN_API_TOKEN"] = "test-token"
        os.environ["GLEAN_ACT_AS"] = "test@example.com"

        # Mock the Glean class where it's directly used
        self.mock_glean_patcher = patch("langchain_glean.retrievers.people.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Create mock entities client
        mock_entities = MagicMock()
        mock_client = MagicMock()
        self.mock_glean.return_value.__enter__.return_value.client = mock_client
        mock_client.entities = mock_entities

        # Create mock sample data
        self.mock_person1 = SimpleNamespace(
            id="person-123",
            name="Jane Doe",
            metadata=SimpleNamespace(
                title="Software Engineer",
                email="jane@example.com",
                department="Engineering",
                location="New York",
                phone="123-456-7890",
            ),
        )

        self.mock_person2 = SimpleNamespace(
            id="person-456",
            name="John Smith",
            metadata=SimpleNamespace(
                title="Product Manager",
                email="john@example.com",
                department="Product",
                location="San Francisco",
                phone="098-765-4321",
            ),
        )

        # Create mock list response
        mock_results = [self.mock_person1, self.mock_person2]
        mock_response = SimpleNamespace(results=mock_results)

        # Mock the list method
        mock_entities.list.return_value = mock_response

        # Mock the list_async method - create a proper MagicMock
        list_async_mock = MagicMock()

        async def list_async_side_effect(*args, **kwargs):
            return mock_response

        list_async_mock.side_effect = list_async_side_effect
        mock_entities.list_async = list_async_mock

        # Initialize the retriever
        self.retriever = GleanPeopleProfileRetriever()

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
        assert self.retriever.k == 10

    def test_init_with_custom_k(self) -> None:
        """Test initialization with a custom k value."""
        retriever = GleanPeopleProfileRetriever(k=5)
        assert retriever.k == 5

    def test_init_with_missing_env_vars(self) -> None:
        """Test initialization with missing environment variables."""
        del os.environ["GLEAN_INSTANCE"]
        del os.environ["GLEAN_API_TOKEN"]

        with pytest.raises(ValueError):
            GleanPeopleProfileRetriever()

    def test_invoke_with_string_query(self) -> None:
        """Test the invoke method with a string query."""
        docs = self.retriever.invoke("software engineer")

        # Verify the entities.list method was called with the correct parameters (SDK 0.11+ uses unpacked kwargs)
        self.mock_glean.return_value.__enter__.return_value.client.entities.list.assert_called_once()
        call_args = self.mock_glean.return_value.__enter__.return_value.client.entities.list.call_args

        # Check the unpacked kwargs
        assert call_args[1]["query"] == "software engineer"

        # Check the documents returned
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "Jane Doe\nSoftware Engineer"
        assert docs[0].metadata["email"] == "jane@example.com"
        assert docs[0].metadata["department"] == "Engineering"
        assert docs[0].metadata["location"] == "New York"

        assert docs[1].page_content == "John Smith\nProduct Manager"
        assert docs[1].metadata["email"] == "john@example.com"
        assert docs[1].metadata["department"] == "Product"

    def test_invoke_with_basic_request(self) -> None:
        """Test the invoke method with a PeopleProfileBasicRequest."""
        request = PeopleProfileBasicRequest(
            query="engineer",
            filters={"department": "Engineering"},
            page_size=5,
        )

        _ = self.retriever.invoke(request)

        # Verify the entities.list method was called with the correct parameters (SDK 0.11+ uses unpacked kwargs)
        self.mock_glean.return_value.__enter__.return_value.client.entities.list.assert_called_once()
        call_args = self.mock_glean.return_value.__enter__.return_value.client.entities.list.call_args

        # Check the unpacked kwargs
        assert call_args[1]["query"] == "engineer"

    def test_invoke_with_filters_only(self) -> None:
        """Test the invoke method with filters but no query."""
        request = PeopleProfileBasicRequest(
            filters={"department": "Engineering"},
            page_size=5,
        )

        _ = self.retriever.invoke(request)

        # Verify the entities.list method was called with the correct parameters (SDK 0.11+ uses unpacked kwargs)
        self.mock_glean.return_value.__enter__.return_value.client.entities.list.assert_called_once()

    def test_invoke_with_native_request(self) -> None:
        """Test the invoke method with a native ListEntitiesRequest."""
        entities_request = ListEntitiesRequest(
            entity_type="PEOPLE",
            query="manager",
            page_size=3,
            filter=[
                FacetFilter(
                    field_name="department",
                    values=[FacetFilterValue(value="Product", relation_type=RelationType.EQUALS)],
                )
            ],
        )

        docs = self.retriever.invoke(entities_request)

        # Verify the entities.list method was called with unpacked request params (SDK 0.11+)
        self.mock_glean.return_value.__enter__.return_value.client.entities.list.assert_called_once()
        call_args = self.mock_glean.return_value.__enter__.return_value.client.entities.list.call_args
        assert call_args[1]["query"] == "manager"

        # Check the documents returned
        assert len(docs) == 2

    async def test_ainvoke_with_string_query(self) -> None:
        """Test the ainvoke method with a string query."""
        docs = await self.retriever.ainvoke("software engineer")

        # Verify the entities.list_async method was called with the correct parameters
        assert self.mock_glean.return_value.__enter__.return_value.client.entities.list_async.called

        # Check the documents returned
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "Jane Doe\nSoftware Engineer"
        assert docs[1].page_content == "John Smith\nProduct Manager"

    def test_with_invalid_request(self) -> None:
        """Test that an invalid request raises ValueError."""
        # Request with no query and no filters should raise ValueError
        with pytest.raises(ValueError):
            PeopleProfileBasicRequest()

        # But either a query or filters is fine
        PeopleProfileBasicRequest(query="test")
        PeopleProfileBasicRequest(filters={"department": "Engineering"})

    def test_error_handling(self) -> None:
        """Test error handling when Glean API call fails."""
        from unittest.mock import MagicMock

        from glean.api_client import errors

        # Simulate a GleanError with required raw_response
        mock_response = MagicMock()
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_glean.return_value.__enter__.return_value.client.entities.list.side_effect = error

        with pytest.raises(ValueError, match="Glean client error"):
            self.retriever.invoke("test query")

        # Simulate a generic exception
        self.mock_glean.return_value.__enter__.return_value.client.entities.list.side_effect = Exception("Generic error")

        # Should return empty list rather than raise for generic exceptions
        docs = self.retriever.invoke("test query")
        assert len(docs) == 0
