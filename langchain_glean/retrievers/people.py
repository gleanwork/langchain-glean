from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from glean import Glean, errors, models
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, Field, model_validator


class PeopleProfileBasicRequest(BaseModel):
    """Basic subset of ``ListEntitiesRequest`` for people search.

    Provides the handful of fields commonly required when interacting with an LLM
    or writing simple code – a free-text ``query`` plus optional ``filters`` and
    ``page_size``.  Supply the full :pyclass:`glean.models.ListEntitiesRequest`
    instead when you need advanced control.
    """

    query: Optional[str] = Field(
        default=None,
        description="Free-text query to search people by name, title, etc.",
    )

    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional mapping of facet-name -> value to filter people (e.g. { 'email': 'jane@acme.com' }).",
    )

    page_size: Optional[int] = Field(
        default=None,
        description="Hint for how many people to return (1-100, default 10).",
        ge=1,
        le=100,
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_query_or_filter(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
        if not values.get("query") and not (values.get("filters") or {}):
            raise ValueError('At least one of "query" or "filters" must be provided.')
        return values


class GleanPeopleProfileRetriever(BaseRetriever):
    """Retriever that queries Glean's people directory and returns LangChain Documents.

    Setup:
        Install ``langchain-glean`` and set environment variables
        ``GLEAN_API_TOKEN`` and ``GLEAN_INSTANCE``. Optionally set ``GLEAN_ACT_AS``
        if using a global token.

        .. code-block:: bash

            pip install -U langchain-glean
            export GLEAN_API_TOKEN="your-api-token"  # Can be a global or user token
            export GLEAN_INSTANCE="your-glean-subdomain"
            export GLEAN_ACT_AS="user@example.com"  # Only required for global tokens

    Example:
        .. code-block:: python

            from langchain_glean.retrievers import GleanPeopleProfileRetriever

            people = GleanPeopleProfileRetriever()  # Will use environment variables

    Usage:
        .. code-block:: python

            query = "engineering manager"

            people.invoke(query)

        .. code-block:: none

            [Document(page_content='Jane Doe\nEngineering Manager',
                     metadata={'email': 'jane@example.com', 'location': '...'}),
             Document(page_content='John Smith\nSenior Engineering Manager',
                     metadata={'email': 'john@example.com', 'location': '...'})]

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"List the engineering managers from the provided information.

            People: {people}

            Format as a bulleted list.\"\"\"
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"people": people | format_docs}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("engineering manager")
    """

    instance: str = Field(description="Glean instance/subdomain (e.g. 'acme')")
    api_token: str = Field(description="Glean API token (user or global)")
    act_as: Optional[str] = Field(
        default=None,
        description="Email to act as when using a global token. Ignored for user tokens.",
    )
    k: Optional[int] = Field(default=10, description="Number of results to return per invocation.")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the required auth environment variables are present.

        Args:
            values: The values to validate.

        Returns:
            The validated values.

        Raises:
            ValueError: If auth credentials are missing from kwargs or environment.
        """
        values = values or {}
        values["instance"] = get_from_dict_or_env(values, "instance", "GLEAN_INSTANCE")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")

        return values

    def __init__(self, **kwargs: Any) -> None:  # noqa: D401
        super().__init__(**kwargs)
        try:
            g = Glean(api_token=self.api_token, instance=self.instance)
            self._client = g.client
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to initialize Glean client: {e}") from e

    def _get_relevant_documents(
        self,
        query: Union[str, PeopleProfileBasicRequest, models.ListEntitiesRequest],
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            entities_req = self._build_entities_request(query, **kwargs)
            response = self._client.entities.list(request=entities_req)
        except errors.GleanError as err:
            raise ValueError(f"Glean client error: {err}") from err
        except Exception:
            # Fallback – return empty results when the SDK call fails (e.g. no network)
            return []

        docs: List[Document] = []
        people_results = getattr(response, "results", None) or []  # type: ignore[attr-defined]

        for person in people_results:  # type: ignore[assignment]
            metadata = {}

            if getattr(person, "metadata", None):
                metadata = {k: v for k, v in person.metadata.__dict__.items() if v}

            name = getattr(person, "name", "Unknown")
            title = metadata.get("title", "")
            page_text = f"{name}\n{title}".strip()

            docs.append(Document(page_content=page_text, metadata=metadata))

        return docs

    async def _aget_relevant_documents(
        self,
        query: Union[str, PeopleProfileBasicRequest, models.ListEntitiesRequest],
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            entities_req = self._build_entities_request(query, **kwargs)
            response = await self._client.entities.list_async(request=entities_req)
        except errors.GleanError as err:
            raise ValueError(f"Glean client error: {err}") from err
        except Exception:
            return []

        docs: List[Document] = []
        people_results = getattr(response, "results", None) or []  # type: ignore[attr-defined]

        for person in people_results:  # type: ignore[assignment]
            metadata = {}

            if getattr(person, "metadata", None):
                metadata = {k: v for k, v in person.metadata.__dict__.items() if v}

            name = getattr(person, "name", "Unknown")
            title = metadata.get("title", "")
            page_text = f"{name}\n{title}".strip()

            docs.append(Document(page_content=page_text, metadata=metadata))

        return docs

    def _build_entities_request(
        self,
        input_val: Union[
            str,
            PeopleProfileBasicRequest,
            models.ListEntitiesRequest,
        ],
        **kwargs: Any,
    ) -> models.ListEntitiesRequest:  # noqa: D401
        """Create a ``ListEntitiesRequest`` for the people directory."""

        if isinstance(input_val, models.ListEntitiesRequest):
            return input_val

        if isinstance(input_val, PeopleProfileBasicRequest):
            data = input_val

            req = models.ListEntitiesRequest(entity_type="PEOPLE")  # type: ignore[arg-type]

            if data.page_size is not None:
                req.page_size = data.page_size
            else:
                req.page_size = self.k or 10

            if data.query:
                req.query = data.query

            if data.filters:
                facets = []
                for field_name, value in data.filters.items():
                    facets.append(
                        models.FacetFilter(
                            field_name=field_name,
                            values=[models.FacetFilterValue(value=value, relation_type=models.RelationType.EQUALS)],
                        )
                    )
                req.filter_ = facets  # Set filter_ instead of filter

            return req

        req = models.ListEntitiesRequest(entity_type="PEOPLE", query=str(input_val))  # type: ignore[arg-type]
        if "page_size" in kwargs:
            req.page_size = kwargs["page_size"]
        else:
            req.page_size = self.k or 10

        return req
