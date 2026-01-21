import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_glean.tools.run_agent import GleanRunAgentTool, RunAgentArgs


class TestGleanRunAgentTool:
    """Test the GleanRunAgentTool class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up the test environment."""
        # Set environment variables for testing
        os.environ["GLEAN_INSTANCE"] = "test-instance"
        os.environ["GLEAN_API_TOKEN"] = "test-api-token"

        # Mock the Glean class where it's used directly in the tool
        self.mock_glean_patcher = patch("langchain_glean.tools.run_agent.Glean")
        self.mock_glean = self.mock_glean_patcher.start()

        # Mock the client property of the Glean instance
        mock_client = MagicMock()
        self.mock_glean.return_value.__enter__.return_value.client = mock_client

        # Create mock agents client
        mock_agents = MagicMock()
        mock_client.agents = mock_agents

        # Configure the run method to return a mock response
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"result": "success", "output": "Mock agent response"}'
        mock_agents.run.return_value = mock_response

        # Configure async method
        async def mock_run_async(*args, **kwargs):
            # Return the same response as the sync method
            return mock_response

        mock_agents.run_async = mock_run_async

        # Initialize the tool
        self.tool = GleanRunAgentTool()

        yield

        # Clean up after tests
        self.mock_glean_patcher.stop()

        # Clean up environment variables after tests
        for var in ["GLEAN_INSTANCE", "GLEAN_API_TOKEN"]:
            os.environ.pop(var, None)

    def test_init(self) -> None:
        """Test the initialization of the tool."""
        assert self.tool.name == "glean_run_agent"
        assert "Run a Glean agent by ID" in self.tool.description
        assert self.tool.args_schema == RunAgentArgs
        assert self.tool.return_direct is True

    def test_run_with_required_params(self) -> None:
        """Test _run with only required parameters."""
        agent_id = "test-agent-id"
        fields = {"input": "Test input"}

        result = self.tool._run(agent_id=agent_id, fields=fields)

        # Verify that run was called with the correct parameters
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.assert_called_once_with(agent_id=agent_id, input=fields)

        assert result == '{"result": "success", "output": "Mock agent response"}'

    def test_run_with_non_json_response(self) -> None:
        """Test _run with a response that doesn't support model_dump_json."""
        agent_id = "test-agent-id"
        fields = {"input": "Test input"}

        # Mock response that doesn't have model_dump_json
        mock_response = "Raw string response"
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.return_value = mock_response

        result = self.tool._run(agent_id=agent_id, fields=fields)

        # Verify that run was called with the correct parameters
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.assert_called_once_with(agent_id=agent_id, input=fields)

        assert result == "Raw string response"

    def test_run_with_glean_error(self) -> None:
        """Test _run when a GleanError occurs."""
        from glean.api_client import errors

        # Mock GleanError with required raw_response
        mock_response = MagicMock()
        mock_response.text = "Raw error response"
        error = errors.GleanError("Test error", raw_response=mock_response)
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.side_effect = error

        result = self.tool._run(agent_id="test-agent-id", fields={})

        assert "Glean API error" in result
        assert "Test error" in result

    def test_run_with_generic_exception(self) -> None:
        """Test _run when a generic exception occurs."""
        # Mock generic exception
        self.mock_glean.return_value.__enter__.return_value.client.agents.run.side_effect = Exception("Generic error")

        result = self.tool._run(agent_id="test-agent-id", fields={})

        assert "Error running agent" in result
        assert "Generic error" in result

    @pytest.mark.asyncio
    async def test_arun_with_required_params(self) -> None:
        """Test _arun with only required parameters."""
        agent_id = "test-agent-id"
        fields = {"input": "Test input"}

        # Override async method for this test
        async def mock_run_async(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.model_dump_json.return_value = '{"result": "success", "output": "Mock agent response"}'
            return mock_response

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async

        result = await self.tool._arun(agent_id=agent_id, fields=fields)

        assert result == '{"result": "success", "output": "Mock agent response"}'

    @pytest.mark.asyncio
    async def test_arun_with_non_json_response(self) -> None:
        """Test _arun with a response that doesn't support model_dump_json."""
        agent_id = "test-agent-id"
        fields = {"input": "Test input"}

        # Override async method for this test
        async def mock_run_async(*args, **kwargs):
            return "Raw string response"

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async

        result = await self.tool._arun(agent_id=agent_id, fields=fields)

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
        async def mock_run_async(*args, **kwargs):
            raise error

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async

        result = await self.tool._arun(agent_id="test-agent-id", fields={})

        assert "Glean API error" in result
        assert "Test error" in result

    @pytest.mark.asyncio
    async def test_arun_with_generic_exception(self) -> None:
        """Test _arun when a generic exception occurs."""

        # Override async method for this test
        async def mock_run_async(*args, **kwargs):
            raise Exception("Generic error")

        self.mock_glean.return_value.__enter__.return_value.client.agents.run_async = mock_run_async

        result = await self.tool._arun(agent_id="test-agent-id", fields={})

        assert "Error running agent" in result
        assert "Generic error" in result
