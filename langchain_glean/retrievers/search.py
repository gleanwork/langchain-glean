# ruff: noqa: I001
from typing import Any, Iterable, List, Optional, Union

from glean.api_client import Glean, errors, models
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, field_validator

from langchain_glean._api_client_mixin import GleanAPIClientMixin


class SearchBasicRequest(BaseModel):
    """Basic subset of ``SearchRequest`` covering the most common fields."""

    query: str = Field(..., description="The search query string.")
    data_sources: Optional[List[str]] = Field(
        default=None,
        description="Optional list of datasource names to restrict the search (e.g. 'github', 'gdrive').",
    )

    @field_validator("data_sources", mode="before")
    @classmethod
    def _clean_data_sources(cls, v: Optional[Iterable[str]]) -> Optional[List[str]]:  # noqa: D401
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        try:
            return [str(s) for s in v]
        except Exception:  # noqa: BLE001
            return None


class GleanSearchRetriever(GleanAPIClientMixin, BaseRetriever):
    """Retriever that uses Glean's search API via the Glean client.

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

            from langchain_glean.retrievers import GleanSearchRetriever

            retriever = GleanSearchRetriever()  # Will use environment variables

    Usage:
        .. code-block:: python

            query = "quarterly sales report"

            retriever.invoke(query)

        .. code-block:: none

            [Document(page_content='Sales increased by 15% in Q2...',
                     metadata={'title': 'Q2 Sales Report', 'url': '...'}),
             Document(page_content='Q1 results showed strong performance...',
                     metadata={'title': 'Q1 Sales Analysis', 'url': '...'})]

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("What were our Q2 sales results?")

        .. code-block:: none

            "Based on the provided context, sales increased by 15% in Q2."
    """

    k: Optional[int] = Field(default=10, description="Number of results to return. Maps to page_size in the Glean API.")

    def _get_relevant_documents(
        self,
        query: Union[str, "SearchBasicRequest", models.SearchRequest],
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant to the query using Glean's search API via the Glean client.

        Args:
            query: The query to search for
            run_manager: The run manager to use for the search
            **kwargs: Additional keyword arguments

        Returns:
            A list of documents relevant to the query
        """
        try:
            search_request = self._build_search_request(query, **kwargs)

            try:
                with Glean(api_token=self.api_token, instance=self.instance) as g:
                    headers = self._http_headers()
                    # Use vars() instead of model_dump() due to SDK's custom serializer
                    params = {k: v for k, v in vars(search_request).items() if not k.startswith("_") and v is not None}
                    response = g.client.search.query(**params, http_headers=headers)

            except errors.GleanError as client_err:
                run_manager.on_retriever_error(Exception(f"Glean client error: {str(client_err)}"))
                return []

            documents = []
            if response and response.results:
                for result in response.results:
                    try:
                        document = self._build_document(result)
                        documents.append(document)
                    except Exception as doc_error:
                        run_manager.on_retriever_error(doc_error)
                        continue

            k_limit = kwargs.get("k") if "k" in kwargs else self.k
            if k_limit is not None and isinstance(k_limit, int):
                documents = documents[:k_limit]

            return documents

        except Exception as e:
            run_manager.on_retriever_error(e)
            return []

    async def _aget_relevant_documents(
        self,
        query: Union[str, "SearchBasicRequest", models.SearchRequest],
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant to the query using Glean's search API via the Glean client (async version).

        Args:
            query: The query to search for
            run_manager: The run manager to use for the search
            **kwargs: Additional keyword arguments

        Returns:
            A list of documents relevant to the query
        """
        try:
            search_request = self._build_search_request(query, **kwargs)

            try:
                with Glean(api_token=self.api_token, instance=self.instance) as g:
                    headers = self._http_headers()
                    # Use vars() instead of model_dump() due to SDK's custom serializer
                    params = {k: v for k, v in vars(search_request).items() if not k.startswith("_") and v is not None}
                    response = await g.client.search.query_async(**params, http_headers=headers)

            except errors.GleanError as client_err:
                await run_manager.on_retriever_error(Exception(f"Glean client error: {str(client_err)}"))
                return []

            documents = []
            if response and response.results:
                for result in response.results:
                    try:
                        document = self._build_document(result)
                        documents.append(document)
                    except Exception as doc_error:
                        await run_manager.on_retriever_error(doc_error)
                        continue

            k_limit = kwargs.get("k") if "k" in kwargs else self.k
            if k_limit is not None and isinstance(k_limit, int):
                documents = documents[:k_limit]

            return documents

        except Exception as e:
            await run_manager.on_retriever_error(e)
            return []

    def _build_search_request(self, query: Union[str, "SearchBasicRequest", models.SearchRequest], **kwargs: Any) -> models.SearchRequest:
        """Build a ``models.SearchRequest`` from either a simple :class:`SearchBasicRequest` *or* the field-by-field kwargs style used today.
        This keeps backwards compatibility while nudging users (and LLMs) toward the *minimal* input schema.

        If ``query`` is a :class:`SearchBasicRequest` instance we ignore ``kwargs`` – everything is
        encoded in that object.
        """

        if isinstance(query, models.SearchRequest):
            return query

        if isinstance(query, SearchBasicRequest):
            data = query  # type: SearchBasicRequest

            sr = models.SearchRequest(query=data.query)

            if data.data_sources:
                sr.request_options = models.SearchRequestOptions(
                    facet_bucket_size=10,
                    facet_filters=[
                        models.FacetFilter(
                            field_name="datasource",
                            values=[models.FacetFilterValue(value=ds, relation_type=models.RelationType.EQUALS) for ds in data.data_sources],
                        )
                    ],
                )

            return sr

        search_request = models.SearchRequest(query=query)  # type: ignore[arg-type]

        if "k" in kwargs and kwargs.get("k") is not None:
            raw_k = kwargs["k"]
            raw_page_size = kwargs.get("page_size", 10)
            search_request.page_size = max(int(raw_k), int(raw_page_size))
        elif self.k is not None:
            search_request.page_size = int(self.k)
        elif "page_size" in kwargs and kwargs["page_size"] is not None:
            search_request.page_size = int(kwargs["page_size"])

        if "max_snippet_size" in kwargs:
            search_request.max_snippet_size = kwargs["max_snippet_size"]
        if "cursor" in kwargs:
            search_request.cursor = kwargs["cursor"]
        if "tracking_token" in kwargs:
            search_request.tracking_token = kwargs["tracking_token"]
        if "timeout_millis" in kwargs:
            search_request.timeout_millis = kwargs["timeout_millis"]

        if "request_options" in kwargs:
            search_request.request_options = kwargs["request_options"]

        return search_request

    def _build_document(self, result: models.SearchResult) -> Document:
        """
        Build a LangChain Document object from a Glean search result.

        Args:
            result: SearchResult object from Glean API

        Returns:
            Document: LangChain Document object built from the result
        """
        snippets = result.snippets if hasattr(result, "snippets") else []
        text_snippets = []

        if snippets:
            for snippet in snippets:
                if hasattr(snippet, "text") and snippet.text:
                    text_snippets.append(str(snippet.text))

        page_content = "\n".join(text_snippets) if text_snippets else ""

        if not page_content.strip():
            page_content = str(result.title) if hasattr(result, "title") and result.title else ""

        document_data = result.document if hasattr(result, "document") else None
        document_id = document_data.id if document_data and hasattr(document_data, "id") else ""

        metadata = {
            "title": result.title if hasattr(result, "title") else "",
            "url": result.url if hasattr(result, "url") else "",
            "source": "glean",
            "document_id": document_id,
            "tracking_token": result.tracking_token if hasattr(result, "tracking_token") else "",
        }

        if document_data:
            if hasattr(document_data, "datasource"):
                metadata["datasource"] = document_data.datasource
            if hasattr(document_data, "doc_type"):
                metadata["doc_type"] = document_data.doc_type

            doc_metadata = document_data.metadata if hasattr(document_data, "metadata") else None
            if doc_metadata:
                for src_field, dest_field in [
                    ("datasourceInstance", "datasource_instance"),
                    ("objectType", "object_type"),
                    ("mimeType", "mime_type"),
                    ("loggingId", "logging_id"),
                    ("visibility", "visibility"),
                    ("documentCategory", "document_category"),
                ]:
                    if hasattr(doc_metadata, src_field) and getattr(doc_metadata, src_field):
                        metadata[dest_field] = getattr(doc_metadata, src_field)

                if hasattr(doc_metadata, "createTime") and doc_metadata.createTime:
                    metadata["create_time"] = doc_metadata.createTime
                if hasattr(doc_metadata, "updateTime") and doc_metadata.updateTime:
                    metadata["update_time"] = doc_metadata.updateTime

                if hasattr(doc_metadata, "author") and doc_metadata.author:
                    author_data = doc_metadata.author
                    if hasattr(author_data, "name"):
                        metadata["author"] = author_data.name
                    if hasattr(author_data, "email"):
                        metadata["author_email"] = author_data.email

                if hasattr(doc_metadata, "interactions") and doc_metadata.interactions:
                    interactions = doc_metadata.interactions
                    if hasattr(interactions, "shares") and interactions.shares:
                        days_ago = interactions.shares[0].num_days_ago if hasattr(interactions.shares[0], "num_days_ago") else 0
                        metadata["shared_days_ago"] = str(days_ago)

        if hasattr(result, "clustered_results") and result.clustered_results is not None:
            metadata["clustered_results_count"] = str(len(result.clustered_results))

        if hasattr(result, "debug_info"):
            metadata["debug_info"] = str(result.debug_info)

        return Document(
            page_content=page_content,
            metadata=metadata,
        )
