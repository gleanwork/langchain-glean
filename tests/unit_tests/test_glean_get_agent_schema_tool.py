import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_glean.tools.get_agent_schema import GleanGetAgentSchemaTool, _GetSchemaArgs


class TestGleanGetAgentSchemaTool:
    """Test the GleanGetAgentSchemaTool class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        # Set environment variables for testing
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-api-token"

        # Mock the Glean class where it's directly used
        self.mock_glean_patcher = patch("langchain_glean.tools.get_agent_schema.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Mock the client property of the Glean instance
        mock_client = MagicMock()
        self.mock_glean.return_value.__enter__.return_value.client = mock_client

        # Create mock agents client
        mock_agents = MagicMock()
        mock_client.agents = mock_agents

        # Create the mock response
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"inputs": [{"name": "input", "type": "STRING", "required": true}]}'

        # Configure the retrieve_schemas method to return a mock response for sync
        mock_agents.retrieve_schemas.return_value = mock_response

        # Configure async method
        async def mock_retrieve_schemas_async(*args, **kwargs):
            # Return the same response as the sync method
            return mock_response

        mock_agents.retrieve_schemas_async = mock_retrieve_schemas_async

        # Initialize the tool
        self.tool = GleanGetAgentSchemaTool()

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN"]:
            os.environ.pop(var, None)

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        assert self.tool.name == "glean_get_agent_schema"
        assert "Fetch the input schema for a Glean agent" in self.tool.description
        assert self.tool.args_schema == _GetSchemaArgs

    def test_run(self) -> None:
        """Test _run method."""
        agent_id = "test-agent-id"

        result = self.tool._run(agent_id=agent_id)

        # Verify that retrieve_schemas was called with the correct parameters
        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas.assert_called_once_with(agent_id=agent_id)

        expected_json = '{"inputs": [{"name": "input", "type": "STRING", "required": true}]}'
        assert result == expected_json

    def test_run_with_non_json_response(self) -> None:
        """Test _run with a response that doesn't support model_dump_json."""
        agent_id = "test-agent-id"

        # Mock response that doesn't have model_dump_json
        mock_response = "Raw string response"
        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas.return_value = mock_response

        result = self.tool._run(agent_id=agent_id)

        # Verify that retrieve_schemas was called with the correct parameters
        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas.assert_called_once_with(agent_id=agent_id)

        assert result == "Raw string response"

    def test_run_with_glean_error(self) -> None:
        """Test _run when a GleanError occurs."""
        from glean.api_client import errors

        # Mock GleanError with required raw_response
        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas.side_effect = error

        result = self.tool._run(agent_id="test-agent-id")

        assert "Glean API error" in result
        assert "Test error" in result

    def test_run_with_generic_exception(self) -> None:
        """Test _run when a generic exception occurs."""
        # Mock generic exception
        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas.side_effect = Exception("Generic error")

        result = self.tool._run(agent_id="test-agent-id")

        assert "Error getting agent schema" in result
        assert "Generic error" in result

    @pytest.mark.asyncio
    async def test_arun(self) -> None:
        """Test _arun method."""
        agent_id = "test-agent-id"

        # Override async method for this test
        async def mock_retrieve_schemas_async(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.model_dump_json.return_value = '{"inputs": [{"name": "input", "type": "STRING", "required": true}]}'
            return mock_response

        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas_async = mock_retrieve_schemas_async

        result = await self.tool._arun(agent_id=agent_id)

        expected_json = '{"inputs": [{"name": "input", "type": "STRING", "required": true}]}'
        assert result == expected_json

    @pytest.mark.asyncio
    async def test_arun_with_non_json_response(self) -> None:
        """Test _arun with a response that doesn't support model_dump_json."""
        agent_id = "test-agent-id"

        # Override async method for this test
        async def mock_retrieve_schemas_async(*args, **kwargs):
            return "Raw string response"

        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas_async = mock_retrieve_schemas_async

        result = await self.tool._arun(agent_id=agent_id)

        assert result == "Raw string response"

    @pytest.mark.asyncio
    async def test_arun_with_glean_error(self) -> None:
        """Test _arun when a GleanError occurs."""
        from glean.api_client import errors

        # Mock GleanError with required raw_response
        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("Test error", raw_response=mock_response)

        # Override async method for this test
        async def mock_retrieve_schemas_async(*args, **kwargs):
            raise error

        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas_async = mock_retrieve_schemas_async

        result = await self.tool._arun(agent_id="test-agent-id")

        assert "Glean API error" in result
        assert "Test error" in result

    @pytest.mark.asyncio
    async def test_arun_with_generic_exception(self) -> None:
        """Test _arun when a generic exception occurs."""

        # Override async method for this test
        async def mock_retrieve_schemas_async(*args, **kwargs):
            raise Exception("Generic error")

        self.mock_glean.return_value.__enter__.return_value.client.agents.retrieve_schemas_async = mock_retrieve_schemas_async

        result = await self.tool._arun(agent_id="test-agent-id")

        assert "Error getting agent schema" in result
        assert "Generic error" in result
