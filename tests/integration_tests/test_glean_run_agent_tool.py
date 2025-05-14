import json
import os
import unittest

from dotenv import load_dotenv

from langchain_glean.tools.run_agent import GleanRunAgentTool


class TestGleanRunAgentToolIntegration(unittest.TestCase):
    """Integration tests for the GleanRunAgentTool."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

        # Define a test agent ID - this should be a valid agent ID in your Glean instance
        self.agent_id = os.environ.get("TEST_AGENT_ID")
        if not self.agent_id:
            self.skipTest("TEST_AGENT_ID not found in environment variables")

        # Initialize the tool
        self.tool = GleanRunAgentTool()

    def test_run_with_basic_input(self) -> None:
        """Test running the tool with basic input."""
        # Basic fields for the agent
        fields = {"input": "What can this agent do?"}

        # Run the tool
        result = self.tool.run(agent_id=self.agent_id, fields=fields)

        # Verify the result is valid JSON
        try:
            response_json = json.loads(result)
            self.assertIsInstance(response_json, dict)
            # The actual structure depends on your agent's response format
        except json.JSONDecodeError:
            # If not JSON, it should be a string response
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_run_with_complex_input(self) -> None:
        """Test running the tool with more complex input."""
        # More complex fields for the agent
        # Adjust these according to your agent's schema
        fields = {"input": "What can this agent do?", "param1": "value1", "param2": "value2"}

        # Run the tool
        result = self.tool.run(agent_id=self.agent_id, fields=fields)

        # Verify the result
        try:
            response_json = json.loads(result)
            self.assertIsInstance(response_json, dict)
        except json.JSONDecodeError:
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_run_with_streaming(self) -> None:
        """Test running the tool with streaming enabled."""
        # Basic fields for the agent
        fields = {"input": "What can this agent do?"}

        # Run the tool with streaming enabled
        result = self.tool.run(agent_id=self.agent_id, fields=fields, stream=True)

        # Verify the result
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    async def test_arun_with_basic_input(self) -> None:
        """Test async running the tool with basic input."""
        # Basic fields for the agent
        fields = {"input": "What can this agent do?"}

        # Run the tool asynchronously
        result = await self.tool.arun(agent_id=self.agent_id, fields=fields)

        # Verify the result
        try:
            response_json = json.loads(result)
            self.assertIsInstance(response_json, dict)
        except json.JSONDecodeError:
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    async def test_arun_with_complex_input(self) -> None:
        """Test async running the tool with more complex input."""
        # More complex fields for the agent
        # Adjust these according to your agent's schema
        fields = {"input": "What can this agent do?", "param1": "value1", "param2": "value2"}

        # Run the tool asynchronously
        result = await self.tool.arun(agent_id=self.agent_id, fields=fields)

        # Verify the result
        try:
            response_json = json.loads(result)
            self.assertIsInstance(response_json, dict)
        except json.JSONDecodeError:
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    async def test_arun_with_streaming(self) -> None:
        """Test async running the tool with streaming enabled."""
        # Basic fields for the agent
        fields = {"input": "What can this agent do?"}

        # Run the tool asynchronously with streaming enabled
        result = await self.tool.arun(agent_id=self.agent_id, fields=fields, stream=True)

        # Verify the result
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
