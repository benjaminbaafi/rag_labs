"""Utility to inspect Azure AI Search index schema."""
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_index_schema():
    """Get the schema of the current search index."""
    cfg = config.get_config()
    credential = AzureKeyCredential(cfg.azure_search_key)
    index_client = SearchIndexClient(
        endpoint=cfg.search_endpoint,
        credential=credential
    )
    
    try:
        index = index_client.get_index(cfg.azure_search_index_name)
        print(f"\nIndex: {index.name}")
        print("=" * 60)
        print("\nFields:")
        print("-" * 60)
        for field in index.fields:
            field_type = field.type
            is_vector = hasattr(field, 'vector_search_dimensions') and field.vector_search_dimensions
            is_searchable = hasattr(field, 'searchable') and field.searchable
            
            print(f"  {field.name}:")
            print(f"    Type: {field_type}")
            if is_vector:
                print(f"    Vector dimensions: {field.vector_search_dimensions}")
                print(f"    Vector profile: {field.vector_search_profile_name}")
            if is_searchable:
                print(f"    Searchable: Yes")
            print()
        
        if hasattr(index, 'vector_search') and index.vector_search:
            print("\nVector Search Configuration:")
            print("-" * 60)
            for profile in index.vector_search.profiles:
                print(f"  Profile: {profile.name}")
                print(f"    Algorithm: {profile.algorithm_configuration_name}")
        
        return index
    except Exception as e:
        print(f"Error getting index schema: {e}")
        return None


if __name__ == "__main__":
    get_index_schema()

