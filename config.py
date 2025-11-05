"""Configuration management for RAG labs using Pydantic."""
from models.config import AzureConfig as _AzureConfig

# Create singleton instance
_config_instance: _AzureConfig | None = None


def get_config() -> _AzureConfig:
    """Get or create AzureConfig instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = _AzureConfig()
        _config_instance.validate()
    return _config_instance


# For backward compatibility, expose the class and instance
AzureConfig = _AzureConfig
config = get_config()

