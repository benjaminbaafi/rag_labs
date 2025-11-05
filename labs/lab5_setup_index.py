"""
Lab 5: Setting Up Azure AI Search Index
========================================

This lab demonstrates how to:
1. Create an Azure AI Search index
2. Upload documents with embeddings
3. Configure the index for RAG

Learning Objectives:
- Understand Azure AI Search index structure
- Learn how to create and populate indexes
- Configure indexes for different search types
"""

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from utils import EmbeddingGenerator
import config
import json


def create_index():
    """Create an Azure AI Search index configured for RAG."""
    print(f"\n{'='*60}")
    print("LAB 5: Setting Up Azure AI Search Index")
    print(f"{'='*60}\n")
    
    cfg = config.get_config()
    credential = AzureKeyCredential(cfg.azure_search_key)
    index_client = SearchIndexClient(
        endpoint=cfg.search_endpoint,
        credential=credential
    )
    
    # Define index fields
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                   vector_search_dimensions=1536, vector_search_profile_name="my-vector-profile")
    ]
    
    # Configure vector search
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="my-hnsw-config"
            )
        ]
    )
    
    # Configure semantic search
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")]
        )
    )
    
    semantic_search = SemanticSearch(configurations=[semantic_config])
    
    # Create index
    cfg = config.get_config()
    index = SearchIndex(
        name=cfg.azure_search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )
    
    try:
        print(f"Creating index: {cfg.azure_search_index_name}...")
        index_client.create_index(index)
        print("✓ Index created successfully!\n")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Index {cfg.azure_search_index_name} already exists.\n")
        else:
            print(f"Error creating index: {e}\n")
            raise


def upload_sample_documents():
    """Upload sample documents with embeddings to the index."""
    print("Uploading sample documents...")
    
    from utils import AzureSearchClient, EmbeddingGenerator
    
    # Sample documents
    sample_docs = [
        {
            "id": "1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
            "category": "AI Basics"
        },
        {
            "id": "2",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. It's particularly effective for image recognition, natural language processing, and speech recognition.",
            "category": "AI Basics"
        },
        {
            "id": "3",
            "title": "Neural Network Architecture",
            "content": "Neural networks consist of interconnected nodes (neurons) organized in layers. The input layer receives data, hidden layers process it, and the output layer produces predictions. Training involves adjusting weights to minimize error.",
            "category": "AI Basics"
        },
        {
            "id": "4",
            "title": "Transformer Models",
            "content": "Transformer models revolutionized NLP with attention mechanisms. They process sequences in parallel rather than sequentially, making them faster and more efficient. Popular examples include BERT, GPT, and T5.",
            "category": "Advanced AI"
        },
        {
            "id": "5",
            "title": "RAG Systems",
            "content": "Retrieval-Augmented Generation combines information retrieval with language generation. It retrieves relevant documents from a knowledge base and uses them as context for generating accurate, grounded responses.",
            "category": "Advanced AI"
        }
    ]
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator()
    print("Generating embeddings for documents...")
    
    contents = [doc["content"] for doc in sample_docs]
    embeddings = embedding_gen.generate_embeddings(contents)
    
    # Add embeddings to documents
    for doc, embedding in zip(sample_docs, embeddings):
        doc["contentVector"] = embedding
    
    # Upload to index
    search_client = AzureSearchClient()
    print("Uploading documents to index...")
    result = search_client.upload_documents(sample_docs)
    
    print(f"✓ Uploaded {len(sample_docs)} documents successfully!\n")
    return sample_docs


def print_index_schema():
    """Print the index schema for reference."""
    print("INDEX SCHEMA:")
    print("-" * 60)
    schema = {
        "fields": [
            {"name": "id", "type": "String (key)"},
            {"name": "title", "type": "String (searchable)"},
            {"name": "content", "type": "String (searchable)"},
            {"name": "category", "type": "String (filterable, facetable)"},
            {"name": "contentVector", "type": "Collection(Single) - 1536 dimensions"}
        ],
        "vector_search": {
            "algorithm": "HNSW",
            "metric": "cosine",
            "profile": "my-vector-profile"
        },
        "semantic_search": {
            "configuration": "default",
            "prioritized_fields": ["title", "content"]
        }
    }
    print(json.dumps(schema, indent=2))
    print()


if __name__ == "__main__":
    print_index_schema()
    
    # Create index
    create_index()
    
    # Upload sample documents
    upload_sample_documents()
    
    print(f"{'='*60}")
    print("Setup complete! You can now run the other labs.")
    print(f"{'='*60}\n")

