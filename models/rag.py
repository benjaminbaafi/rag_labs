"""Pydantic models for RAG operations."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from models.search import Document, SearchResult


class RAGRequest(BaseModel):
    """Request model for RAG operations."""
    
    query: str = Field(
        ...,
        min_length=1,
        description="User query/question"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of documents to retrieve"
    )
    search_type: str = Field(
        default="keyword",
        description="Search type: keyword, vector, hybrid, or semantic"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation"
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=4000,
        description="Maximum tokens to generate"
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt (optional)"
    )
    use_multi_step: bool = Field(
        default=False,
        description="Use multi-step retrieval"
    )
    use_context_management: bool = Field(
        default=False,
        description="Use context window management"
    )
    
    @property
    def is_hybrid(self) -> bool:
        """Check if hybrid search is requested."""
        return self.search_type.lower() == "hybrid"
    
    @property
    def is_vector(self) -> bool:
        """Check if vector search is requested."""
        return self.search_type.lower() in ["vector", "hybrid"]


class RAGMetadata(BaseModel):
    """Metadata about the RAG operation."""
    
    num_documents_retrieved: int = Field(..., description="Number of documents retrieved")
    search_type: str = Field(..., description="Type of search performed")
    search_scores: List[float] = Field(default_factory=list, description="Search relevance scores")
    retrieval_time: Optional[float] = Field(None, description="Time taken for retrieval (seconds)")
    generation_time: Optional[float] = Field(None, description="Time taken for generation (seconds)")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    context_tokens: Optional[int] = Field(None, description="Tokens in context")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGResponse(BaseModel):
    """Response model for RAG operations."""
    
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    retrieved_documents: List[Document] = Field(
        default_factory=list,
        description="Retrieved documents"
    )
    search_results: Optional[List[SearchResult]] = Field(
        None,
        description="Full search results with scores"
    )
    metadata: RAGMetadata = Field(
        default_factory=lambda: RAGMetadata(num_documents_retrieved=0, search_type="unknown"),
        description="RAG operation metadata"
    )
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the response."""
        if key == "num_documents_retrieved":
            self.metadata.num_documents_retrieved = value
        elif key == "search_type":
            self.metadata.search_type = value
        elif key == "top_k":
            self.metadata.additional_info["top_k"] = value
        elif key == "temperature":
            self.metadata.additional_info["temperature"] = value
        else:
            self.metadata.additional_info[key] = value
    
    def get_answer_only(self) -> str:
        """Get only the answer text, without any metadata."""
        return self.answer
    
    def get_metadata_summary(self) -> str:
        """Get a formatted summary of metadata."""
        lines = [
            f"Documents Retrieved: {self.metadata.num_documents_retrieved}",
            f"Search Type: {self.metadata.search_type}"
        ]
        
        if self.metadata.search_scores:
            avg_score = sum(self.metadata.search_scores) / len(self.metadata.search_scores)
            lines.append(f"Average Relevance Score: {avg_score:.4f}")
        
        if self.metadata.retrieval_time:
            lines.append(f"Retrieval Time: {self.metadata.retrieval_time:.2f}s")
        
        if self.metadata.generation_time:
            lines.append(f"Generation Time: {self.metadata.generation_time:.2f}s")
        
        if self.metadata.total_tokens:
            lines.append(f"Total Tokens: {self.metadata.total_tokens}")
        
        return "\n".join(lines)
