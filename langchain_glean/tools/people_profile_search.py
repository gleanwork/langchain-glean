"""Tool wrapper around :class:`~langchain_glean.retrievers.people.PeopleProfileRetriever`."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from langchain_glean.retrievers.people import (
    GleanPeopleProfileRetriever,
    PeopleProfileBasicRequest,
)


class GleanPeopleProfileSearchTool(BaseTool):
    """Tool for searching Glean's people directory."""

    name: str = "people_profile_search"
    description: str = "Search for people within the organization by name, email, or other keywords."

    args_schema: type = PeopleProfileBasicRequest

    retriever: GleanPeopleProfileRetriever
    return_direct: bool = False

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Run the tool synchronously supporting multiple invocation patterns."""
        try:
            if args and not kwargs:
                tool_input = args[0]
            elif "input" in kwargs:
                tool_input = kwargs["input"]
            else:
                tool_input = PeopleProfileBasicRequest(**kwargs) if kwargs else None

            if tool_input is None:
                return "Error: No valid input provided"

            docs = self.retriever.invoke(tool_input)  # type: ignore[arg-type]

            if not docs:
                return "No matching people found.\n- "

            results: list[str] = []
            for doc in docs:
                name = doc.page_content.split("\n")[0].strip()
                title = doc.metadata.get("title", "").strip()
                email = doc.metadata.get("email", "").strip()
                department = doc.metadata.get("department", "").strip()
                location = doc.metadata.get("location", "").strip()

                person_info: list[str] = [f"- {name}"]
                if title:
                    person_info.append(f"  Title: {title}")
                if email:
                    person_info.append(f"  Email: {email}")
                if department:
                    person_info.append(f"  Department: {department}")
                if location:
                    person_info.append(f"  Location: {location}")

                results.append("\n".join(person_info))

            return "\n\n".join(results)
        except Exception as e:  # noqa: BLE001
            return f"Error searching people profiles: {str(e)}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Async variant supporting the same flexible calling conventions."""
        try:
            if args and not kwargs:
                tool_input = args[0]
            elif "input" in kwargs:
                tool_input = kwargs["input"]
            else:
                tool_input = PeopleProfileBasicRequest(**kwargs) if kwargs else None

            if tool_input is None:
                return "Error: No valid input provided"

            docs = await self.retriever.ainvoke(tool_input)  # type: ignore[arg-type]

            if not docs:
                return "No matching people found.\n- "

            results: list[str] = []
            for doc in docs:
                name = doc.page_content.split("\n")[0].strip()
                title = doc.metadata.get("title", "").strip()
                email = doc.metadata.get("email", "").strip()
                department = doc.metadata.get("department", "").strip()
                location = doc.metadata.get("location", "").strip()

                person_info: list[str] = [f"- {name}"]
                if title:
                    person_info.append(f"  Title: {title}")
                if email:
                    person_info.append(f"  Email: {email}")
                if department:
                    person_info.append(f"  Department: {department}")
                if location:
                    person_info.append(f"  Location: {location}")

                results.append("\n".join(person_info))

            return "\n\n".join(results)
        except Exception as e:  # noqa: BLE001
            return f"Error searching people profiles: {str(e)}"
