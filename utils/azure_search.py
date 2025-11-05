"""Azure AI Search utilities for RAG labs."""
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from typing import List, Dict, Any, Optional
import config
from models.search import SearchResult, SearchResponse, Document


class AzureSearchClient:
    """Wrapper for Azure AI Search operations."""
    
    def __init__(self):
        """Initialize Azure AI Search client."""
        cfg = config.get_config()
        cfg.validate()
        
        credential = AzureKeyCredential(cfg.azure_search_key)
        self.client = SearchClient(
            endpoint=cfg.search_endpoint,
            index_name=cfg.azure_search_index_name,
            credential=credential
        )
    
    def search_keyword(self, query: str, top: int = 5, use_pydantic: bool = True) -> List[Dict[str, Any]] | SearchResponse:
        """
        Perform keyword search.
        
        Args:
            query: Search query string
            top: Number of results to return
            use_pydantic: If True, return Pydantic models; if False, return dicts
            
        Returns:
            List of search results or SearchResponse model
        """
        results = self.client.search(
            search_text=query,
            top=top,
            include_total_count=True
        )
        
        if use_pydantic:
            search_results = self._convert_to_search_results(list(results), query, "keyword")
            return search_results
        return [dict(result) for result in results]
    
    def search_semantic(
        self, 
        query: str, 
        top: int = 5,
        use_pydantic: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using Azure AI Search semantic ranking.
        
        Semantic search uses AI-powered ranking to understand query intent
        and rank results by relevance. It doesn't require embeddings.
        
        Args:
            query: Search query string
            top: Number of results to return
            use_pydantic: If True, return Pydantic models; if False, return dicts
            
        Returns:
            List of search results or SearchResponse model
        """
        results = self.client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="default",
            top=top,
            query_caption="extractive",
            query_answer="extractive",
            include_total_count=True
        )
        
        if use_pydantic:
            search_results = self._convert_to_search_results(list(results), query, "semantic")
            return search_results
        return [dict(result) for result in results]
    
    def search_vector(
        self, 
        vector: List[float], 
        top: int = 5,
        fields: str = "content_vector"  # Changed from "contentVector" to match index
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            vector: Query embedding vector
            top: Number of results to return
            fields: Name of the vector field to search
            
        Returns:
            List of search results
        """
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top,
            fields=fields
        )
        
        results = self.client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top
        )
        return [dict(result) for result in results]
    
    def search_hybrid(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        top: int = 5,
        vector_fields: str = "content_vector"  # Changed from "contentVector" to match index
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (keyword + vector).
        
        Args:
            query: Keyword search query
            vector: Optional query embedding vector
            top: Number of results to return
            vector_fields: Name of the vector field to search
            
        Returns:
            List of search results
        """
        vector_queries = []
        if vector:
            vector_queries.append(
                VectorizedQuery(
                    vector=vector,
                    k_nearest_neighbors=top,
                    fields=vector_fields
                )
            )
        
        results = self.client.search(
            search_text=query,
            vector_queries=vector_queries,
            top=top,
            query_type="semantic" if not vector_queries else None,
            semantic_configuration_name="default" if not vector_queries else None
        )
        return [dict(result) for result in results]
    
    def upload_documents(self, documents: List[Dict[str, Any]]):
        """
        Upload documents to the search index.
        
        Args:
            documents: List of documents to upload
        """
        result = self.client.upload_documents(documents=documents)
        return result
    
    def _convert_to_search_results(
        self, 
        raw_results: List[Dict[str, Any]], 
        query: str, 
        search_type: str
    ) -> SearchResponse:
        """Convert raw search results to Pydantic models."""
        search_results = []
        total_count = None
        
        for result_dict in raw_results:
            # Extract score
            score = result_dict.get("@search.score", 0.0)
            reranker_score = result_dict.get("@search.reranker_score")
            
            # Extract document fields
            doc_id = result_dict.get("id", "")
            doc_content = result_dict.get("content", result_dict.get("text", ""))
            doc_title = result_dict.get("title")
            doc_category = result_dict.get("category")
            doc_vector = result_dict.get("contentVector")
            
            # Create Document
            document = Document(
                id=doc_id,
                title=doc_title,
                content=doc_content,
                category=doc_category,
                content_vector=doc_vector,
                metadata={k: v for k, v in result_dict.items() 
                         if k not in ["id", "title", "content", "text", "category", "contentVector", "content_vector", "@search.score", "@search.reranker_score"]}
            )
            
            # Create SearchResult
            search_result = SearchResult(
                document=document,
                score=score,
                reranker_score=reranker_score,
                metadata={k: v for k, v in result_dict.items() 
                         if k.startswith("@search.")}
            )
            search_results.append(search_result)
            
            # Extract total count if available
            if "@odata.count" in result_dict:
                total_count = result_dict["@odata.count"]
        
        return SearchResponse(
            results=search_results,
            total_count=total_count,
            query=query,
            search_type=search_type
        )

