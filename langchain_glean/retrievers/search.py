import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from langchain_glean.client import GleanAuth, GleanClient
from langchain_glean.client.glean_client import GleanClientError, GleanConnectionError, GleanHTTPError

DEFAULT_PAGE_SIZE = 100


class GleanSearchParameters(BaseModel):
    """Parameters for Glean search API."""

    query: str = Field(..., description="The search query to execute")
    cursor: Optional[str] = Field(default=None, description="Pagination cursor for retrieving more results")
    disable_spellcheck: Optional[bool] = Field(default=None, description="Whether to disable spellcheck")
    max_snippet_size: Optional[int] = Field(default=None, description="Maximum number of characters for snippets")
    page_size: Optional[int] = Field(default=None, description="Number of results to return per page")
    result_tab_ids: Optional[List[str]] = Field(default=None, description="IDs of result tabs to fetch results for")
    timeout_millis: Optional[int] = Field(default=None, description="Timeout in milliseconds for the request")
    tracking_token: Optional[str] = Field(default=None, description="Token for tracking related requests")
    request_options: Optional[Dict[str, Any]] = Field(default=None, description="Additional request options including facet filters")

    def to_dict(self) -> Dict[str, Any]:
        result = {k: v for k, v in self.model_dump().items() if v is not None}

        camel_case_result = {}
        for key, value in result.items():
            if key == "page_size":
                camel_case_result["pageSize"] = value
            elif key == "disable_spellcheck":
                camel_case_result["disableSpellcheck"] = value
            elif key == "max_snippet_size":
                camel_case_result["maxSnippetSize"] = value
            elif key == "result_tab_ids":
                camel_case_result["resultTabIds"] = value
            elif key == "timeout_millis":
                camel_case_result["timeoutMillis"] = value
            elif key == "tracking_token":
                camel_case_result["trackingToken"] = value
            elif key == "request_options":
                camel_case_result["requestOptions"] = value
            else:
                camel_case_result[key] = value

        return camel_case_result


class GleanSearchRetriever(BaseRetriever):
    """Retriever that uses Glean's search API via the Glean client.

    Setup:
        Install ``langchain-glean`` and set environment variables
        ``GLEAN_API_TOKEN`` and ``GLEAN_SUBDOMAIN``. Optionally set ``GLEAN_ACT_AS``
        if using a global token.

        .. code-block:: bash

            pip install -U langchain-glean
            export GLEAN_API_TOKEN="your-api-token"  # Can be a global or user token
            export GLEAN_SUBDOMAIN="your-glean-subdomain"
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

    subdomain: str = Field(description="Subdomain for Glean instance")
    api_token: str = Field(description="API token for Glean")
    act_as: Optional[str] = Field(
        default=None, description="Email for the user to act as. Required only when using a global token, not needed for user tokens."
    )
    k: Optional[int] = Field(default=None, description="Number of results to return. Maps to page_size in the Glean API.")

    _auth: GleanAuth = PrivateAttr()
    _client: GleanClient = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and subdomain exists in environment.

        Args:
            values: The values to validate.

        Returns:
            The validated values.

        Raises:
            ValueError: If api key or subdomain are not found in environment.
        """
        values = values or {}
        values["subdomain"] = get_from_dict_or_env(values, "subdomain", "GLEAN_SUBDOMAIN")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")

        return values

    def __init__(self) -> None:
        """Initialize the retriever.

        All required values are pulled from environment variables during model validation.
        """
        super().__init__()

        try:
            self._auth = GleanAuth(api_token=self.api_token, subdomain=self.subdomain, act_as=self.act_as)
            self._client = GleanClient(auth=self._auth)
        except Exception as e:
            raise ValueError(f"Failed to initialize Glean client: {str(e)}")

    def _build_search_params(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Build the search parameters dictionary for the Glean API request.

        Args:
            query: The query to search for
            **kwargs: Additional keyword arguments that can include any parameters from GleanSearchParameters

        Returns:
            A dictionary containing the search parameters in the format expected by the Glean API
        """
        search_params: Dict[str, Any] = {"query": query}

        # The k parameter is a langchain parameter that maps to the number of documents to return
        if "k" in kwargs:
            # Use k as the minimum page size, but might request more to ensure we get enough results
            search_params["page_size"] = max(kwargs.get("k"), kwargs.get("page_size", DEFAULT_PAGE_SIZE))
        elif self.k is not None:
            # Use self.k as the minimum page size
            search_params["page_size"] = max(self.k, kwargs.get("page_size", DEFAULT_PAGE_SIZE))
        elif "page_size" not in kwargs:
            # Default page size if nothing is specified
            search_params["page_size"] = DEFAULT_PAGE_SIZE

        for key, value in kwargs.items():
            search_params[key] = value

        params = GleanSearchParameters(
            query=search_params["query"],
            page_size=search_params.get("page_size"),
            cursor=search_params.get("cursor"),
            disable_spellcheck=search_params.get("disable_spellcheck"),
            max_snippet_size=search_params.get("max_snippet_size"),
            result_tab_ids=search_params.get("result_tab_ids"),
            timeout_millis=search_params.get("timeout_millis"),
            tracking_token=search_params.get("tracking_token"),
            request_options=search_params.get("request_options"),
        )
        return params.to_dict()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        """Get documents relevant to the query using Glean's search API via the Glean client.

        Args:
            query: The query to search for
            run_manager: The run manager to use for the search
            **kwargs: Additional keyword arguments that can include any parameters from GleanSearchParameters

        Returns:
            A list of documents relevant to the query
        """

        try:
            payload = self._build_search_params(query, **kwargs)

            try:
                response = self._client.post("search", data=json.dumps(payload), headers={"Content-Type": "application/json"})
                search_results = self._client.parse_response(response)

            except GleanHTTPError as http_err:
                error_details = f"HTTP Error {http_err.status_code}"
                if http_err.response:
                    error_details += f": {http_err.response}"
                run_manager.on_retriever_error(Exception(f"Glean API error: {error_details}"))
                return []
            except GleanConnectionError as conn_err:
                run_manager.on_retriever_error(Exception(f"Glean connection error: {str(conn_err)}"))
                return []
            except GleanClientError as client_err:
                run_manager.on_retriever_error(Exception(f"Glean client error: {str(client_err)}"))
                return []

            documents = []
            for result in search_results.get("results", []):
                try:
                    document = self._build_document(result)
                    documents.append(document)
                except Exception as doc_error:
                    run_manager.on_retriever_error(doc_error)
                    continue

            # Limit the number of documents based on the k parameter
            k_limit = kwargs.get("k") if "k" in kwargs else self.k
            if k_limit is not None and isinstance(k_limit, int):
                documents = documents[:k_limit]

            return documents

        except Exception as e:
            run_manager.on_retriever_error(e)
            return []

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        """Get documents relevant to the query using Glean's search API via the Glean client (async version).

        Args:
            query: The query to search for
            run_manager: The run manager to use for the search
            **kwargs: Additional keyword arguments that can include any parameters from GleanSearchParameters

        Returns:
            A list of documents relevant to the query
        """
        try:
            payload = self._build_search_params(query, **kwargs)

            import asyncio

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(
                    None, lambda: self._client.post("search", data=json.dumps(payload), headers={"Content-Type": "application/json"})
                )
                search_results = self._client.parse_response(response)
            except GleanHTTPError as http_err:
                error_details = f"HTTP Error {http_err.status_code}"
                if http_err.response:
                    error_details += f": {http_err.response}"
                await run_manager.on_retriever_error(Exception(f"Glean API error: {error_details}"))
                return []
            except GleanConnectionError as conn_err:
                await run_manager.on_retriever_error(Exception(f"Glean connection error: {str(conn_err)}"))
                return []
            except GleanClientError as client_err:
                await run_manager.on_retriever_error(Exception(f"Glean client error: {str(client_err)}"))
                return []

            documents = []
            for result in search_results.get("results", []):
                try:
                    document = self._build_document(result)
                    documents.append(document)
                except Exception as doc_error:
                    await run_manager.on_retriever_error(doc_error)
                    continue

            return documents

        except Exception as e:
            await run_manager.on_retriever_error(e)
            return []

    def _build_document(self, result: Dict[str, Any]) -> Document:
        """
        Build a LangChain Document object from a Glean search result.

        Args:
            result: Dictionary containing search result data from Glean API

        Returns:
            Document: LangChain Document object built from the result
        """
        snippets = result.get("snippets", [])
        text_snippets = []

        for snippet in snippets:
            snippet_text = snippet.get("text", "")
            if snippet_text:
                text_snippets.append(snippet_text)

        page_content = "\n".join(text_snippets) if text_snippets else ""

        if not page_content.strip():
            page_content = result.get("title", "")

        document_data = result.get("document", {})
        document_id = document_data.get("id", "")

        metadata = {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "source": "glean",
            "document_id": document_id,
            "tracking_token": result.get("trackingToken", ""),
        }

        if document_data:
            metadata.update(
                {
                    "datasource": document_data.get("datasource", ""),
                    "doc_type": document_data.get("docType", ""),
                }
            )

            doc_metadata = document_data.get("metadata", {})
            if doc_metadata:
                metadata.update(
                    {
                        "datasource_instance": doc_metadata.get("datasourceInstance", ""),
                        "object_type": doc_metadata.get("objectType", ""),
                        "mime_type": doc_metadata.get("mimeType", ""),
                        "logging_id": doc_metadata.get("loggingId", ""),
                        "visibility": doc_metadata.get("visibility", ""),
                        "document_category": doc_metadata.get("documentCategory", ""),
                    }
                )

                if "createTime" in doc_metadata:
                    metadata["create_time"] = doc_metadata["createTime"]
                if "updateTime" in doc_metadata:
                    metadata["update_time"] = doc_metadata["updateTime"]

                if "author" in doc_metadata:
                    author_data = doc_metadata["author"]
                    metadata["author"] = author_data.get("name", "")
                    metadata["author_email"] = author_data.get("email", "")

                if "interactions" in doc_metadata:
                    interactions = doc_metadata["interactions"]
                    if "shares" in interactions:
                        metadata["shared_days_ago"] = interactions["shares"][0].get("numDaysAgo", 0) if interactions["shares"] else 0

        if "clusteredResults" in result:
            metadata["clustered_results_count"] = len(result["clusteredResults"])

        if "debugInfo" in result:
            metadata["debug_info"] = str(result["debugInfo"])

        return Document(
            page_content=page_content,
            metadata=metadata,
        )
