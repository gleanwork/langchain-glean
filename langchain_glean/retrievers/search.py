import json
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Optional
from pydantic import Field, PrivateAttr

from langchain_glean.client import GleanAuth, GleanClient

class GleanSearchRetriever(BaseRetriever):
    """Retriever that uses Glean's search API via the Glean client."""
    
    subdomain: str = Field(description="Subdomain for Glean instance (e.g., 'scio-prod')")
    api_key: str = Field(description="API key for Glean")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    act_as: Optional[str] = Field(default=None, description="Email to act as when making requests to Glean")
    
    _auth: GleanAuth = PrivateAttr()
    _client: GleanClient = PrivateAttr()
    
    def __init__(
        self, 
        subdomain: str,
        api_key: str,
        max_results: int = 5,
        act_as: Optional[str] = None
    ):
        """Initialize the GleanRetriever.
        
        Args:
            subdomain: Subdomain for Glean instance (e.g., 'scio-prod')
            api_key: API key for Glean
            max_results: Maximum number of results to return
            act_as: Email to act as when making requests to Glean
        """
        super().__init__(
            subdomain=subdomain,
            api_key=api_key,
            max_results=max_results,
            act_as=act_as
        )
        
        try:
            self._auth = GleanAuth(
                api_token=self.api_key,
                subdomain=self.subdomain,
                act_as=self.act_as
            )
            self._client = GleanClient(auth=self._auth)
            
        except Exception as e:
            raise ValueError(f"Failed to initialize Glean client: {str(e)}")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query using Glean's search API via the Glean client."""
        try:
            payload = {
                "query": query,
                "pageSize": self.max_results
            }
            
            search_results = self._client.post(
                "search", 
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            documents = []
            
            for result in search_results.get("results", []):
                content = ""
                for snippet in result.get("snippets", []):
                    content += snippet.get("snippet", "") + "\n"
                
                if not content.strip():
                    content = result.get("title", "")
                
                metadata = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": "glean"
                }
                
                if "metadata" in result:
                    result_metadata = result["metadata"]
                    metadata.update({
                        "datasource": result_metadata.get("datasource", ""),
                        "document_id": result_metadata.get("documentId", ""),
                        "mime_type": result_metadata.get("mimeType", ""),
                        "object_type": result_metadata.get("objectType", ""),
                        "container": result_metadata.get("container", "")
                    })
                    
                    if "author" in result_metadata:
                        metadata["author"] = result_metadata["author"].get("name", "")
                    
                    if "createTime" in result_metadata:
                        metadata["create_time"] = result_metadata["createTime"]
                    if "updateTime" in result_metadata:
                        metadata["update_time"] = result_metadata["updateTime"]
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            return documents
        
        except Exception as e:
            print(f"Error retrieving documents from Glean: {str(e)}")
            return [] 