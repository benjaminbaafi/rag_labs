"""LLM utilities for RAG labs."""
from openai import AzureOpenAI, OpenAI
from typing import List, Dict, Any
import config
from models.search import Document, SearchResult
from models.rag import RAGRequest, RAGResponse
from prompts.loader import load_prompt, load_template


class LLMClient:
    """Wrapper for LLM operations."""
    
    def __init__(self):
        """Initialize LLM client - uses Azure OpenAI if available, falls back to OpenAI."""
        cfg = config.get_config()
        cfg.validate()
        
        # Prefer Azure OpenAI if configured, otherwise use standard OpenAI
        if cfg.uses_azure_openai:
            self.client = AzureOpenAI(
                api_key=cfg.azure_openai_api_key,
                api_version=cfg.openai_api_version,
                azure_endpoint=cfg.openai_endpoint
            )
            self.deployment = cfg.azure_openai_deployment_name
            self.provider = "azure_openai"
        elif cfg.uses_openai:
            self.client = OpenAI(api_key=cfg.openai_api_key)
            self.deployment = cfg.openai_deployment_name or "gpt-4"
            self.provider = "openai"
        else:
            raise ValueError(
                "No OpenAI provider configured. Please set either Azure OpenAI or OpenAI credentials."
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context (default: loads from prompts.json)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        # Load default system prompt if not provided
        if system_prompt is None:
            try:
                system_prompt = load_prompt("system.default")
            except FileNotFoundError:
                system_prompt = "You are a helpful assistant."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]] | List[Document],
        system_prompt: str = None
    ) -> str:
        """
        Generate response with retrieved context (RAG pattern).
        
        Args:
            query: User query
            context: List of retrieved documents (dict or Document models)
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated response with context
        """
        # Load RAG system prompt if not provided
        if system_prompt is None:
            try:
                system_prompt = load_prompt("system.rag")
            except FileNotFoundError:
                system_prompt = (
                    "You are a helpful assistant that answers questions based on "
                    "the provided context. If the context doesn't contain enough "
                    "information to answer the question, say so."
                )
        
        # Build context string from retrieved documents
        context_text = "\n\n".join([
            f"Document {i+1}:\n{self._extract_content(doc)}"
            for i, doc in enumerate(context)
        ])
        
        # Load RAG user prompt template
        try:
            prompt = load_template("user.rag", context_text=context_text, query=query)
        except FileNotFoundError:
            prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
        
        return self.generate(prompt, system_prompt=system_prompt)
    
    def _extract_content(self, doc: Dict[str, Any] | Document) -> str:
        """Extract content from document (dict or Document model)."""
        if isinstance(doc, Document):
            return doc.content
        return doc.get('content', doc.get('text', ''))
    
    def generate_rag_response(
        self,
        request: RAGRequest,
        retrieved_documents: List[Document],
        search_results: List[SearchResult] = None
    ) -> RAGResponse:
        """
        Generate RAG response using Pydantic models.
        
        Args:
            request: RAG request model
            retrieved_documents: List of retrieved documents
            search_results: Optional full search results with scores
            
        Returns:
            RAG response model
        """
        # Load system prompt from request or defaults
        if request.system_prompt:
            system_prompt = request.system_prompt
        else:
            try:
                system_prompt = load_prompt("system.rag")
            except FileNotFoundError:
                system_prompt = (
                    "You are a helpful assistant that answers questions based on "
                    "the provided context. If the context doesn't contain enough "
                    "information to answer the question, say so."
                )
        
        answer = self.generate_with_context(
            request.query,
            retrieved_documents,
            system_prompt=system_prompt
        )
        
        response = RAGResponse(
            answer=answer,
            query=request.query,
            retrieved_documents=retrieved_documents,
            search_results=search_results
        )
        
        response.add_metadata("top_k", request.top_k)
        response.add_metadata("search_type", request.search_type)
        response.add_metadata("temperature", request.temperature)
        
        return response

