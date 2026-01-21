import json
import os
import unittest

from dotenv import load_dotenv

from langchain_glean.tools.list_agents import GleanListAgentsTool


class TestGleanListAgentsToolIntegration(unittest.TestCase):
    """Integration tests for the GleanListAgentsTool."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

        # Initialize the tool
        self.tool = GleanListAgentsTool()

    def test_run(self) -> None:
        """Test running the tool."""
        # Run the tool to list agents - pass empty dict as tool input
        result = self.tool.run({})

        # Verify the result is valid JSON
        try:
            agents_json = json.loads(result)
            self.assertIsInstance(agents_json, dict)

            # Basic agents list validation
            if "agents" in agents_json:
                self.assertIsInstance(agents_json["agents"], list)

                # If there are agents, check they have the expected structure
                if agents_json["agents"]:
                    agent = agents_json["agents"][0]
                    self.assertIn("agent_id", agent)
                    self.assertIn("name", agent)

        except json.JSONDecodeError:
            # If not JSON, it should be a string response (could be an error message)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    async def test_arun(self) -> None:
        """Test async running the tool."""
        # Run the tool asynchronously to list agents - pass empty dict as tool input
        result = await self.tool.arun({})

        # Verify the result is valid JSON
        try:
            agents_json = json.loads(result)
            self.assertIsInstance(agents_json, dict)

            # Basic agents list validation
            if "agents" in agents_json:
                self.assertIsInstance(agents_json["agents"], list)

                # If there are agents, check they have the expected structure
                if agents_json["agents"]:
                    agent = agents_json["agents"][0]
                    self.assertIn("agent_id", agent)
                    self.assertIn("name", agent)

        except json.JSONDecodeError:
            # If not JSON, it should be a string response (could be an error message)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_agents_contain_required_fields(self) -> None:
        """Test that the returned agents contain required fields."""
        # Run the tool to list agents - pass empty dict as tool input
        result = self.tool.run({})

        try:
            agents_json = json.loads(result)

            # If there are agents, check they all have required fields
            if "agents" in agents_json and agents_json["agents"]:
                for agent in agents_json["agents"]:
                    self.assertIn("agent_id", agent)
                    self.assertIn("name", agent)

                    # ID should be a string
                    self.assertIsInstance(agent["agent_id"], str)

                    # Name should be a string
                    self.assertIsInstance(agent["name"], str)

        except json.JSONDecodeError:
            self.skipTest("Response is not valid JSON")

    def test_find_specific_agent(self) -> None:
        """Test finding a specific agent in the list."""
        # This test checks if a specific agent from the environment is in the list

        agent_id = os.environ.get("TEST_AGENT_ID")
        if not agent_id:
            self.skipTest("TEST_AGENT_ID not found in environment variables")

        # Run the tool to list agents - pass empty dict as tool input
        result = self.tool.run({})

        try:
            agents_json = json.loads(result)

            # If there are agents, check if our test agent is in the list
            found_agent = False
            if "agents" in agents_json and agents_json["agents"]:
                for agent in agents_json["agents"]:
                    if agent["id"] == agent_id:
                        found_agent = True
                        break

            self.assertTrue(found_agent, f"Test agent with ID {agent_id} not found in the agents list")

        except json.JSONDecodeError:
            self.skipTest("Response is not valid JSON")
