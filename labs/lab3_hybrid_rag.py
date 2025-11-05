"""
Lab 3: Hybrid RAG (Keyword + Vector Search)
============================================

This lab demonstrates hybrid search combining keyword and vector search:
1. Generate embedding for query
2. Perform hybrid search (keyword + vector)
3. Generate answer with retrieved context

Learning Objectives:
- Understand hybrid search benefits
- Learn when to use hybrid vs. single method
- Combine keyword matching with semantic similarity
"""

from utils import AzureSearchClient, LLMClient, EmbeddingGenerator
from utils.display import display_rag_answer, extract_metadata_from_results


def hybrid_rag(query: str, top_k: int = 3):
    """
    Perform RAG using hybrid search (keyword + vector).
    
    Args:
        query: User question
        top_k: Number of documents to retrieve
    """
    print(f"\n{'='*60}")
    print("LAB 3: Hybrid RAG (Keyword + Vector)")
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
    
    # Step 3: Perform hybrid search
    print(f"Step 2: Performing hybrid search (keyword + vector, top {top_k})...")
    try:
        results = search_client.search_hybrid(
            query=query,
            vector=query_embedding,
            top=top_k
        )
    except Exception as e:
        print(f"⚠ Hybrid search not available (vector fields not configured): {str(e)[:100]}")
        print("Falling back to keyword search...")
        results = search_client.search_keyword(query, top=top_k, use_pydantic=False)
    
    if not results:
        print("No documents found. Please ensure your search index is populated.")
        return
    
    # Handle both dict and Pydantic models
    if isinstance(results, list):
        metadata = extract_metadata_from_results(results, search_type="hybrid")
        context_docs = results
    else:
        # SearchResponse model
        metadata = extract_metadata_from_results(results.results, search_type="hybrid")
        context_docs = [result.document for result in results.results]
        results = results.results
    
    # Step 4: Generate answer
    print("Step 3: Generating answer using LLM with retrieved context...")
    answer = llm_client.generate_with_context(query, context_docs)
    
    # Display answer with metadata separated
    display_rag_answer(answer, metadata=metadata, retrieved_docs=results)
    
    return answer


def compare_all_methods(query: str, top_k: int = 3):
    """
    Compare keyword, vector, and hybrid search methods.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve for each method
    """
    print(f"\n{'='*60}")
    print("COMPREHENSIVE COMPARISON: Keyword vs. Vector vs. Hybrid")
    print(f"{'='*60}")
    print(f"\nQuery: {query}\n")
    
    search_client = AzureSearchClient()
    embedding_gen = EmbeddingGenerator()
    query_embedding = embedding_gen.generate_embedding(query)
    
    # Keyword search
    print("KEYWORD SEARCH:")
    print("-" * 60)
    keyword_results = search_client.search_keyword(query, top=top_k, use_pydantic=False)
    for i, result in enumerate(keyword_results, 1):
        content = result.get('content', result.get('text', ''))[:150]
        score = result.get('@search.score', 'N/A')
        print(f"{i}. [Score: {score}] {content}...")
    print()
    
    # Vector search
    print("VECTOR SEARCH:")
    print("-" * 60)
    try:
        vector_results = search_client.search_vector(query_embedding, top=top_k)
        for i, result in enumerate(vector_results, 1):
            content = result.get('content', result.get('text', ''))[:150]
            score = result.get('@search.score', 'N/A')
            print(f"{i}. [Score: {score}] {content}...")
    except Exception as e:
        print(f"⚠ Vector search not available: {str(e)[:100]}")
    print()
    
    # Hybrid search
    print("HYBRID SEARCH:")
    print("-" * 60)
    try:
        hybrid_results = search_client.search_hybrid(query, query_embedding, top=top_k)
        for i, result in enumerate(hybrid_results, 1):
            content = result.get('content', result.get('text', ''))[:150]
            score = result.get('@search.score', 'N/A')
            print(f"{i}. [Score: {score}] {content}...")
    except Exception as e:
        print(f"⚠ Hybrid search not available: {str(e)[:100]}")
    print()
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Interactive mode
    print("\n" + "="*60)
    print("LAB 3: Hybrid RAG - Interactive Mode")
    print("="*60)
    
    query = input("\nEnter your question: ").strip()
    if not query:
        query = "Explain deep learning architectures"
        print(f"Using default query: {query}")
    
    # Run hybrid RAG
    hybrid_rag(query)
    
    # Compare all methods
    compare = input("\nCompare all search methods? (y/n): ").strip().lower()
    if compare == 'y':
        compare_all_methods(query)

