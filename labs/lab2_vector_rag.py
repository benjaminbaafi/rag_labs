"""
Lab 2: Vector-Based RAG
=======================

This lab demonstrates RAG using vector embeddings for semantic search:
1. Generate embedding for query
2. Perform vector similarity search
3. Generate answer with retrieved context

Learning Objectives:
- Understand vector embeddings and semantic search
- Learn how to use embeddings for better retrieval
- Compare keyword vs. vector search results
"""

from utils import AzureSearchClient, LLMClient, EmbeddingGenerator
from utils.display import display_rag_answer, extract_metadata_from_results


def vector_rag(query: str, top_k: int = 3):
    """
    Perform RAG using vector embeddings.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve
    """
    print(f"\n{'='*60}")
    print("LAB 2: Vector-Based RAG")
    print(f"{'='*60}")
    print(f"\nQuery: {query}\n")
    
    # Step 1: Initialize clients
    search_client = AzureSearchClient()
    llm_client = LLMClient()
    embedding_gen = EmbeddingGenerator()
    
    # Step 2: Generate query embedding
    print("Step 1: Generating embedding for query...")
    query_embedding = embedding_gen.generate_embedding(query)
    print(f"Embedding dimension: {len(query_embedding)}\n")
    
    # Step 3: Perform vector search
    print(f"Step 2: Performing vector similarity search (top {top_k})...")
    try:
        results = search_client.search_vector(query_embedding, top=top_k)
    except Exception as e:
        print(f"⚠ Vector search not available: {e}")
        print("Falling back to keyword search...")
        results = search_client.search_keyword(query, top=top_k, use_pydantic=False)
    
    if not results:
        print("No documents found. Please ensure your search index is populated.")
        return
    
    # Handle both Pydantic models and dict results
    if isinstance(results, list):
        metadata = extract_metadata_from_results(results, search_type="vector")
        context_docs = results
    else:
        # SearchResponse model
        metadata = extract_metadata_from_results(results.results, search_type="vector")
        context_docs = [result.document for result in results.results]
        results = results.results
    
    # Step 4: Generate answer
    print("Step 3: Generating answer using LLM with retrieved context...")
    answer = llm_client.generate_with_context(query, context_docs)
    
    # Display answer with metadata separated
    display_rag_answer(answer, metadata=metadata, retrieved_docs=results)
    
    return answer


def compare_keyword_vs_vector(query: str, top_k: int = 3):
    """
    Compare keyword search vs. vector search results.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve for each method
    """
    print(f"\n{'='*60}")
    print("COMPARISON: Keyword vs. Vector Search")
    print(f"{'='*60}")
    print(f"\nQuery: {query}\n")
    
    search_client = AzureSearchClient()
    embedding_gen = EmbeddingGenerator()
    
    # Keyword search
    print("KEYWORD SEARCH:")
    print("-" * 60)
    keyword_results = search_client.search_keyword(query, top=top_k, use_pydantic=False)
    for i, result in enumerate(keyword_results, 1):
        content = result.get('content', result.get('text', ''))
        print(f"{i}. {content[:150]}...")
    
    print("\n" + "=" * 60 + "\n")
    
    # Vector search
    print("VECTOR SEARCH:")
    print("-" * 60)
    query_embedding = embedding_gen.generate_embedding(query)
    try:
        vector_results = search_client.search_vector(query_embedding, top=top_k)
        for i, result in enumerate(vector_results, 1):
            content = result.get('content', result.get('text', ''))
            print(f"{i}. {content[:150]}...")
    except Exception as e:
        print(f"⚠ Vector search not available: {str(e)[:100]}")
        print("(Vector search requires vector fields in your index)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Interactive mode
    print("\n" + "="*60)
    print("LAB 2: Vector-Based RAG - Interactive Mode")
    print("="*60)
    
    query = input("\nEnter your question: ").strip()
    if not query:
        query = "How does neural network training work?"
        print(f"Using default query: {query}")
    
    # Run vector RAG
    vector_rag(query)
    
    # Compare methods
    compare = input("\nCompare keyword vs. vector search? (y/n): ").strip().lower()
    if compare == 'y':
        compare_keyword_vs_vector(query)

