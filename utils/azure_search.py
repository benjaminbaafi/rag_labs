"""Azure AI Search utilities for RAG labs."""
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from typing import List, Dict, Any, Optional
import config
from models.search import SearchResult, SearchResponse, Document


class AzureSearchClient:
    """Wrapper for Azure AI Search operations."""

    ID_FIELD = "chunk_id"
    TITLE_FIELD = "title"
    CONTENT_FIELD = "chunk"
    VECTOR_FIELD = "text_vector"

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

        raw_results = [dict(result) for result in results]
        normalized_results = [self._normalize_result_dict(result) for result in raw_results]

        if use_pydantic:
            search_results = self._convert_to_search_results(normalized_results, query, "keyword")
            return search_results
        return normalized_results
    
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
        
        raw_results = [dict(result) for result in results]
        normalized_results = [self._normalize_result_dict(result) for result in raw_results]

        if use_pydantic:
            search_results = self._convert_to_search_results(normalized_results, query, "semantic")
            return search_results
        return normalized_results
    
    def search_vector(
        self, 
        vector: List[float], 
        top: int = 5,
        fields: str = VECTOR_FIELD
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
        raw_results = [dict(result) for result in results]
        return [self._normalize_result_dict(result) for result in raw_results]
    
    def search_hybrid(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        top: int = 5,
        vector_fields: str = VECTOR_FIELD
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
        raw_results = [dict(result) for result in results]
        return [self._normalize_result_dict(result) for result in raw_results]
    
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
            doc_id = result_dict.get("id", result_dict.get(self.ID_FIELD, ""))
            doc_content = result_dict.get("content", result_dict.get(self.CONTENT_FIELD, result_dict.get("text", "")))
            doc_title = result_dict.get("title", result_dict.get(self.TITLE_FIELD))
            doc_category = result_dict.get("category")
            doc_vector = result_dict.get("contentVector", result_dict.get("content_vector", result_dict.get(self.VECTOR_FIELD)))
            
            # Create Document
            document = Document(
                id=doc_id,
                title=doc_title,
                content=doc_content,
                category=doc_category,
                content_vector=doc_vector,
                metadata={k: v for k, v in result_dict.items() 
                         if k not in [
                             "id",
                             self.ID_FIELD,
                             "title",
                             self.TITLE_FIELD,
                             "content",
                             self.CONTENT_FIELD,
                             "text",
                             "category",
                             "contentVector",
                             "content_vector",
                             self.VECTOR_FIELD,
                             "@search.score",
                             "@search.reranker_score"
                         ]}
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

    def _normalize_result_dict(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw Azure Search result to expected field names."""
        normalized = dict(result_dict)

        if self.ID_FIELD in result_dict:
            normalized.setdefault("id", result_dict[self.ID_FIELD])
        else:
            normalized.setdefault("id", result_dict.get("id", ""))

        content_value = result_dict.get(self.CONTENT_FIELD)
        if content_value is None:
            content_value = result_dict.get("content", result_dict.get("text", ""))
        normalized["content"] = content_value

        if self.TITLE_FIELD in result_dict and "title" not in normalized:
            normalized["title"] = result_dict[self.TITLE_FIELD]

        vector_value = result_dict.get(self.VECTOR_FIELD)
        if vector_value is not None:
            normalized.setdefault("contentVector", vector_value)
            normalized.setdefault("content_vector", vector_value)

        return normalized

