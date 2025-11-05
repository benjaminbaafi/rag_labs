"""Pydantic models for search results."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Document(BaseModel):
    """Document model for search results."""
    
    id: str = Field(..., description="Document ID")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(..., description="Document content")
    category: Optional[str] = Field(None, description="Document category")
    content_vector: Optional[List[float]] = Field(
        None,
        alias="contentVector",
        description="Document embedding vector"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document metadata"
    )
    
    class Config:
        populate_by_name = True


class SearchResult(BaseModel):
    """Search result model with scoring."""
    
    document: Document = Field(..., description="Retrieved document")
    score: float = Field(
        ...,
        alias="@search.score",
        description="Search relevance score"
    )
    reranker_score: Optional[float] = Field(
        None,
        alias="@search.reranker_score",
        description="Reranker score (if applicable)"
    )
    highlights: Optional[List[str]] = Field(
        None,
        description="Highlighted text snippets"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search metadata"
    )
    
    class Config:
        populate_by_name = True
    
    @property
    def content(self) -> str:
        """Get document content."""
        # Access document field - Pydantic resolves this to actual value at runtime
        doc = getattr(self, 'document', None)
        if doc is None:
            return ''
        if isinstance(doc, Document):
            return doc.content  # type: ignore[attr-defined]
        if isinstance(doc, dict):
            return str(doc.get('content', ''))  # type: ignore[attr-defined]
        return ''
    
    @property
    def title(self) -> Optional[str]:
        """Get document title."""
        # Access document field - Pydantic resolves this to actual value at runtime
        doc = getattr(self, 'document', None)
        if doc is None:
            return None
        if isinstance(doc, Document):
            return doc.title  # type: ignore[attr-defined]
        if isinstance(doc, dict):
            return doc.get('title')  # type: ignore[attr-defined]
        return None


class SearchResponse(BaseModel):
    """Search response model."""
    
    results: List[SearchResult] = Field(
        ...,
        description="List of search results"
    )
    total_count: Optional[int] = Field(
        None,
        alias="@odata.count",
        description="Total number of matching documents"
    )
    query: str = Field(..., description="Original search query")
    search_type: str = Field(
        ...,
        description="Type of search performed (keyword, vector, hybrid)"
    )
    
    class Config:
        populate_by_name = True
    
    def get_documents(self) -> List[Document]:
        """Get list of documents from results."""
        return [result.document for result in self.results]
    
    def get_top_k(self, k: int) -> List[Document]:
        """Get top K documents."""
        return [result.document for result in self.results[:k]]

