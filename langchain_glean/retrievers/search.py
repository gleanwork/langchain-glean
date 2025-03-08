import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, PrivateAttr

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
        """Convert parameters to a dictionary for the API request."""
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
        ``GLEAN_API_TOKEN`` and ``GLEAN_SUBDOMAIN``.

        .. code-block:: bash

            pip install -U langchain-glean
            export GLEAN_API_TOKEN="your-api-token"
            export GLEAN_SUBDOMAIN="your-glean-subdomain"

    Key init args:
        subdomain: str
            Subdomain for Glean instance (e.g., 'my-glean')
        api_token: str
            API token for Glean
        act_as: Optional[str]
            Email for the user to act as when making requests to Glean

    Instantiate:
        .. code-block:: python

            from langchain_glean.retrievers import GleanSearchRetriever

            retriever = GleanSearchRetriever(
                subdomain="my-glean",
                api_token="your-api-token",
            )

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

    subdomain: str = Field(description="Subdomain for Glean instance (e.g., 'my-glean')")
    api_token: str = Field(description="API token for Glean")
    act_as: Optional[str] = Field(default=None, description="Email for the user to act as when making requests to Glean")

    _auth: GleanAuth = PrivateAttr()
    _client: GleanClient = PrivateAttr()

    def __init__(self, subdomain: str, api_token: str, act_as: Optional[str] = None) -> None:
        """Initialize the GleanRetriever.

        Args:
            subdomain: Subdomain for Glean instance (e.g., 'my-glean')
            api_token: API token for Glean
            act_as: Email for the user to act as when making requests to Glean
        """

        super().__init__(subdomain=subdomain, api_token=api_token, act_as=act_as)

        try:
            self._auth = GleanAuth(api_token=self.api_token, subdomain=self.subdomain, act_as=self.act_as)
            self._client = GleanClient(auth=self._auth)

        except Exception as e:
            raise ValueError(f"Failed to initialize Glean client: {str(e)}")

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
            search_params = {"query": query}

            if "page_size" not in kwargs:
                search_params["page_size"] = DEFAULT_PAGE_SIZE

            search_params.update(kwargs)

            params = GleanSearchParameters(**search_params)
            payload = params.to_dict()

            try:
                search_results = self._client.post("search", data=json.dumps(payload), headers={"Content-Type": "application/json"})
            except GleanHTTPError as http_err:
                error_details = f"HTTP Error {http_err.status_code}"
                if http_err.response:
                    error_details += f": {http_err.response}"
                run_manager.on_retriever_error(f"Glean API error: {error_details}")
                return []
            except GleanConnectionError as conn_err:
                run_manager.on_retriever_error(f"Glean connection error: {str(conn_err)}")
                return []
            except GleanClientError as client_err:
                run_manager.on_retriever_error(f"Glean client error: {str(client_err)}")
                return []

            documents = []
            for result in search_results.get("results", []):
                try:
                    document = self._build_document(result)
                    documents.append(document)
                except Exception as doc_error:
                    run_manager.on_retriever_error(f"Error processing document: {str(doc_error)}")
                    continue

            return documents

        except Exception as e:
            run_manager.on_retriever_error(f"Error during retrieval: {str(e)}")
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
            search_params = {"query": query}

            if "page_size" not in kwargs:
                search_params["page_size"] = DEFAULT_PAGE_SIZE

            search_params.update(kwargs)

            params = GleanSearchParameters(**search_params)
            payload = params.to_dict()

            import asyncio

            loop = asyncio.get_event_loop()
            try:
                search_results = await loop.run_in_executor(
                    None, lambda: self._client.post("search", data=json.dumps(payload), headers={"Content-Type": "application/json"})
                )
            except GleanHTTPError as http_err:
                error_details = f"HTTP Error {http_err.status_code}"
                if http_err.response:
                    error_details += f": {http_err.response}"
                await run_manager.on_retriever_error(f"Glean API error: {error_details}")
                return []
            except GleanConnectionError as conn_err:
                await run_manager.on_retriever_error(f"Glean connection error: {str(conn_err)}")
                return []
            except GleanClientError as client_err:
                await run_manager.on_retriever_error(f"Glean client error: {str(client_err)}")
                return []

            documents = []
            for result in search_results.get("results", []):
                try:
                    document = self._build_document(result)
                    documents.append(document)
                except Exception as doc_error:
                    await run_manager.on_retriever_error(f"Error processing document: {str(doc_error)}")
                    continue

            return documents

        except Exception as e:
            await run_manager.on_retriever_error(f"Error during retrieval: {str(e)}")
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
