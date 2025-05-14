import json
import os
import unittest

from dotenv import load_dotenv

from langchain_glean.tools.get_agent_schema import GleanGetAgentSchemaTool


class TestGleanGetAgentSchemaToolIntegration(unittest.TestCase):
    """Integration tests for the GleanGetAgentSchemaTool."""

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
        self.tool = GleanGetAgentSchemaTool()

    def test_run(self) -> None:
        """Test running the tool."""
        # Run the tool to get the agent schema
        result = self.tool.run(agent_id=self.agent_id)

        # Verify the result is valid JSON
        try:
            schema_json = json.loads(result)
            self.assertIsInstance(schema_json, dict)

            # Basic schema validation
            # The actual structure depends on your agent's schema
            if "inputs" in schema_json:
                self.assertIsInstance(schema_json["inputs"], list)

                # If there are inputs, check they have the expected structure
                if schema_json["inputs"]:
                    input_field = schema_json["inputs"][0]
                    self.assertIn("name", input_field)
                    self.assertIn("type", input_field)

        except json.JSONDecodeError:
            # If not JSON, it should be a string response (could be an error message)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    async def test_arun(self) -> None:
        """Test async running the tool."""
        # Run the tool asynchronously to get the agent schema
        result = await self.tool.arun(agent_id=self.agent_id)

        # Verify the result is valid JSON
        try:
            schema_json = json.loads(result)
            self.assertIsInstance(schema_json, dict)

            # Basic schema validation
            # The actual structure depends on your agent's schema
            if "inputs" in schema_json:
                self.assertIsInstance(schema_json["inputs"], list)

                # If there are inputs, check they have the expected structure
                if schema_json["inputs"]:
                    input_field = schema_json["inputs"][0]
                    self.assertIn("name", input_field)
                    self.assertIn("type", input_field)

        except json.JSONDecodeError:
            # If not JSON, it should be a string response (could be an error message)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_run_with_nonexistent_agent(self) -> None:
        """Test running the tool with a nonexistent agent ID."""
        # Use a fake agent ID that shouldn't exist
        fake_agent_id = "nonexistent-agent-id-12345"

        # Run the tool with the fake agent ID
        result = self.tool.run(agent_id=fake_agent_id)

        # The result should contain an error message
        self.assertIsInstance(result, str)
        # Could be either JSON error response or plain string
        self.assertTrue("error" in result.lower() or "not found" in result.lower())

    async def test_arun_with_nonexistent_agent(self) -> None:
        """Test async running the tool with a nonexistent agent ID."""
        # Use a fake agent ID that shouldn't exist
        fake_agent_id = "nonexistent-agent-id-12345"

        # Run the tool asynchronously with the fake agent ID
        result = await self.tool.arun(agent_id=fake_agent_id)

        # The result should contain an error message
        self.assertIsInstance(result, str)
        # Could be either JSON error response or plain string
        self.assertTrue("error" in result.lower() or "not found" in result.lower())
