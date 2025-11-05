"""Pydantic models for configuration management."""
from pydantic import Field, field_validator, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AzureConfig(BaseSettings):
    """Azure AI Search and OpenAI configuration with Pydantic validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Azure AI Search
    azure_search_endpoint: HttpUrl = Field(
        ...,
        alias="AZURE_SEARCH_ENDPOINT",
        description="Azure AI Search service endpoint URL"
    )
    azure_search_key: str = Field(
        ...,
        alias="AZURE_SEARCH_KEY",
        description="Azure AI Search admin key",
        min_length=1
    )
    azure_search_index_name: str = Field(
        default="rag-labs-index",
        alias="AZURE_SEARCH_INDEX_NAME",
        description="Azure AI Search index name"
    )
    
    # Azure OpenAI (preferred - use if available)
    azure_openai_endpoint: Optional[HttpUrl] = Field(
        default=None,
        alias="AZURE_OPENAI_ENDPOINT",
        description="Azure OpenAI endpoint URL"
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_API_KEY",
        description="Azure OpenAI API key"
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None,
        alias="AZURE_OPENAI_DEPLOYMENT_NAME",
        description="Azure OpenAI deployment name"
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-ada-002",
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        description="Azure OpenAI embedding deployment name"
    )
    
    # Standard OpenAI (fallback - use if Azure OpenAI not configured)
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI API key (use if Azure OpenAI not configured)"
    )
    openai_api_version: str = Field(
        default="2024-02-15-preview",
        alias="OPENAI_API_VERSION",
        description="Azure OpenAI API version (or OpenAI API version)"
    )
    openai_deployment_name: Optional[str] = Field(
        default="gpt-4",
        alias="OPENAI_DEPLOYMENT_NAME",
        description="OpenAI deployment/model name (fallback)"
    )
    
    @field_validator("azure_search_endpoint", mode="before")
    @classmethod
    def validate_search_endpoint(cls, v):
        """Convert string to HttpUrl if needed."""
        if isinstance(v, str) and not v.startswith("http"):
            return f"https://{v}"
        return v
    
    @field_validator("azure_openai_endpoint", mode="before")
    @classmethod
    def validate_openai_endpoint(cls, v):
        """Convert string to HttpUrl if needed."""
        if isinstance(v, str) and v and not v.startswith("http"):
            return f"https://{v}"
        return v
    
    @property
    def search_endpoint(self) -> str:
        """Get search endpoint as string."""
        return str(self.azure_search_endpoint)
    
    @property
    def openai_endpoint(self) -> Optional[str]:
        """Get Azure OpenAI endpoint as string."""
        return str(self.azure_openai_endpoint) if self.azure_openai_endpoint else None
    
    @property
    def uses_azure_openai(self) -> bool:
        """Check if Azure OpenAI is configured."""
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key and self.azure_openai_deployment_name)
    
    @property
    def uses_openai(self) -> bool:
        """Check if standard OpenAI is configured."""
        return bool(self.openai_api_key)
    
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        # Pydantic validates on instantiation, but we can add custom checks
        if not self.azure_search_endpoint:
            raise ValueError("AZURE_SEARCH_ENDPOINT is required")
        if not self.azure_search_key:
            raise ValueError("AZURE_SEARCH_KEY is required")
        
        # At least one OpenAI provider must be configured
        has_azure_openai = bool(self.azure_openai_endpoint and self.azure_openai_api_key and self.azure_openai_deployment_name)
        has_openai = bool(self.openai_api_key)
        
        if not has_azure_openai and not has_openai:
            raise ValueError(
                "Either AZURE_OPENAI_* credentials or OPENAI_API_KEY must be provided. "
                "Please configure either Azure OpenAI or standard OpenAI."
            )
        
        # If Azure OpenAI is configured, validate all required fields
        if has_azure_openai:
            if not self.azure_openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required when using Azure OpenAI")
            if not self.azure_openai_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required when using Azure OpenAI")
            if not self.azure_openai_deployment_name:
                raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is required when using Azure OpenAI")
        
        return True

