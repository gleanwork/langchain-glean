from typing import Any, Dict, List, Optional

from glean import Glean, errors, models
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, model_validator


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

    instance: str = Field(description="Instance for Glean")
    api_token: str = Field(description="API token for Glean")
    act_as: Optional[str] = Field(
        default=None, description="Email for the user to act as. Required only when using a global token, not needed for user tokens."
    )
    k: Optional[int] = Field(default=None, description="Number of results to return. Maps to page_size in the Glean API.")

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
        values["instance"] = get_from_dict_or_env(values, "instance", "GLEAN_INSTANCE")
        values["api_token"] = get_from_dict_or_env(values, "api_token", "GLEAN_API_TOKEN")
        values["act_as"] = get_from_dict_or_env(values, "act_as", "GLEAN_ACT_AS", default="")

        return values

    def __init__(self) -> None:
        """Initialize the retriever.

        All required values are pulled from environment variables during model validation.
        """
        super().__init__()

        try:
            g = Glean(
                api_token=self.api_token,
                instance=self.instance,
            )
            self._client = g.client
        except Exception as e:
            raise ValueError(f"Failed to initialize Glean client: {str(e)}")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> List[Document]:
        """Get documents relevant to the query using Glean's search API via the Glean client.

        Args:
            query: The query to search for
            run_manager: The run manager to use for the search
            **kwargs: Additional keyword arguments

        Returns:
            A list of documents relevant to the query
        """
        try:
            # Convert keyword arguments to a proper SearchRequest object
            search_request = self._build_search_request(query, **kwargs)

            try:
                response = self._client.search.execute(search_request=search_request)

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
            **kwargs: Additional keyword arguments

        Returns:
            A list of documents relevant to the query
        """
        try:
            # Convert keyword arguments to a proper SearchRequest object
            search_request = self._build_search_request(query, **kwargs)

            try:
                response = await self._client.search.execute_async(search_request=search_request)

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

            # Limit the number of documents based on the k parameter
            k_limit = kwargs.get("k") if "k" in kwargs else self.k
            if k_limit is not None and isinstance(k_limit, int):
                documents = documents[:k_limit]

            return documents

        except Exception as e:
            await run_manager.on_retriever_error(e)
            return []

    def _build_search_request(self, query: str, **kwargs: Any) -> models.SearchRequest:
        """Build a SearchRequest object from query and kwargs.

        Args:
            query: The query to search for
            **kwargs: Additional parameters for the search

        Returns:
            A SearchRequest object for the Glean API
        """
        # Start with query
        params = {"query": query}

        # Handle k parameter (for LangChain compatibility)
        if "k" in kwargs:
            params["page_size"] = max(kwargs.get("k"), kwargs.get("page_size", 100))
        elif self.k is not None:
            params["page_size"] = max(self.k, kwargs.get("page_size", 100))
        elif "page_size" in kwargs:
            params["page_size"] = kwargs["page_size"]

        # Add remaining parameters
        for key, value in kwargs.items():
            if key != "k" and key not in params:
                params[key] = value

        # Create SearchRequest object
        return models.SearchRequest(**params)

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
