import os
import unittest
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_glean.retrievers import GleanSearchRetriever
from langchain_glean.retrievers.people import GleanPeopleProfileRetriever, PeopleProfileBasicRequest
from langchain_glean.tools.chat import GleanChatTool
from langchain_glean.tools.people_profile_search import GleanPeopleProfileSearchTool


class TestEndToEndWorkflows(unittest.TestCase):
    """Test end-to-end workflows combining multiple langchain-glean components."""

    def setUp(self) -> None:
        """Set up test environment variables."""
        super().setUp()

        load_dotenv(override=True)

        if not os.environ.get("GLEAN_INSTANCE") or not os.environ.get("GLEAN_API_TOKEN"):
            self.skipTest("Glean credentials not found in environment variables")

    def test_search_to_chat_workflow(self) -> None:
        """Test workflow combining search retriever with chat tool."""

        retriever = GleanSearchRetriever()
        search_query = "Glean search features"
        docs = retriever.invoke(search_query)

        self.assertIsInstance(docs, List)

        if not docs:
            self.skipTest("No search results found for query: " + search_query)

        self.assertIsInstance(docs[0], Document)

        context = [doc.page_content for doc in docs[:2]]

        chat_tool = GleanChatTool()
        chat_message = "Summarize the key features mentioned in these documents"

        result = chat_tool.invoke(message=chat_message, context=context)

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_people_search_to_chat_workflow(self) -> None:
        """Test workflow combining people search with chat."""
        people_retriever = GleanPeopleProfileRetriever()
        people_request = PeopleProfileBasicRequest(query="engineer", page_size=3)

        people_docs = people_retriever.invoke(people_request)

        self.assertIsInstance(people_docs, List)

        if not people_docs:
            self.skipTest("No people found for query: engineer")

        self.assertIsInstance(people_docs[0], Document)

        people_info = [f"{doc.page_content} - {doc.metadata.get('email', 'No email')}" for doc in people_docs]

        chat_tool = GleanChatTool()
        chat_message = "Who are these people and what roles do they have?"

        result = chat_tool.invoke(message=chat_message, context=people_info)

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_combined_search_people_chat_workflow(self) -> None:
        """Test a more complex workflow combining search, people search, and chat."""
        search_retriever = GleanSearchRetriever()
        project_docs = search_retriever.invoke("project roadmap", page_size=2)

        people_tool = GleanPeopleProfileSearchTool(retriever=GleanPeopleProfileRetriever())
        people_info = people_tool.invoke(PeopleProfileBasicRequest(query="product manager", page_size=2))

        combined_context = []

        if project_docs:
            combined_context.append("Project information:")
            for doc in project_docs:
                combined_context.append(doc.page_content)

        combined_context.append("\nRelevant team members:")
        combined_context.append(people_info)

        chat_tool = GleanChatTool()
        result = chat_tool.invoke(
            message="Based on the project roadmap and team information, provide a summary of the project status and key team members.", context=combined_context
        )

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
