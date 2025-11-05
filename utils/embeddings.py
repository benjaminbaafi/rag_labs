"""Embedding generation utilities for RAG labs."""
from openai import AzureOpenAI, OpenAI
from typing import List
import config


class EmbeddingGenerator:
    """Generate embeddings for text using OpenAI."""
    
    def __init__(self):
        """Initialize embedding generator - uses Azure OpenAI if available, falls back to OpenAI."""
        cfg = config.get_config()
        cfg.validate()
        
        # Prefer Azure OpenAI if configured, otherwise use standard OpenAI
        if cfg.uses_azure_openai:
            self.client = AzureOpenAI(
                api_key=cfg.azure_openai_api_key,
                api_version=cfg.openai_api_version,
                azure_endpoint=cfg.openai_endpoint
            )
            self.deployment = cfg.azure_openai_embedding_deployment
            self.provider = "azure_openai"
        elif cfg.uses_openai:
            self.client = OpenAI(api_key=cfg.openai_api_key)
            self.deployment = "text-embedding-ada-002"
            self.provider = "openai"
        else:
            raise ValueError(
                "No OpenAI provider configured. Please set either Azure OpenAI or OpenAI credentials."
            )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.deployment,
            input=text
        )
        return response.data[0].embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.deployment,
            input=texts
        )
        return [item.embedding for item in response.data]

