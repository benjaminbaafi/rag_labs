"""
Lab 6: Semantic Search RAG
===========================

This lab demonstrates semantic search using Azure AI Search's semantic ranking.
Semantic search uses AI-powered ranking to understand query intent and rank results
by relevance, without requiring embeddings.

Learning Objectives:
- Understand semantic search vs. vector search vs. keyword search
- Learn how to use Azure AI Search semantic ranking
- Compare semantic search with other search methods
- Understand when to use semantic search

Key Concepts:
- Semantic ranking (AI-powered relevance scoring)
- No embeddings required (uses AI to understand intent)
- Query understanding and result ranking
"""

from utils import AzureSearchClient, LLMClient, EmbeddingGenerator
from utils.display import display_rag_answer, extract_metadata_from_results
from typing import List


def semantic_rag(query: str, top_k: int = 3):
    """
    Perform RAG using semantic search.
    
    Semantic search uses AI-powered ranking to understand query intent
    and rank results by relevance. It doesn't require embeddings.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve
    """
    print(f"\n{'='*60}")
    print("LAB 6: Semantic Search RAG")
    print(f"{'='*60}")
    print(f"\nQuery: {query}\n")
    
    # Initialize clients
    search_client = AzureSearchClient()
    llm_client = LLMClient()
    
    # Step 1: Perform semantic search
    print(f"{'='*60}")
    print(f"STEP 1: Semantic Search (top {top_k})")
    print(f"{'='*60}")
    print("Semantic search uses AI to understand query intent and rank results.")
    print("No embeddings required - Azure AI Search handles the semantic understanding.\n")
    
    try:
        results = search_client.search_semantic(query, top=top_k)
    except Exception as e:
        print(f"⚠ Semantic search not available: {str(e)[:150]}")
        print("This may be because:")
        print("  1. Semantic search is not enabled on your Azure AI Search service")
        print("  2. Your index doesn't have semantic search configured")
        print("  3. Your Azure AI Search tier doesn't support semantic search")
        print("\nFalling back to keyword search...\n")
        results = search_client.search_keyword(query, top=top_k, use_pydantic=False)
    
    if not results:
        print("No documents found. Please ensure your search index is populated.")
        return
    
    # Extract metadata
    metadata = extract_metadata_from_results(results, search_type="semantic")
    context_docs = results
    
    # Step 2: Generate answer with context
    print(f"\n{'='*60}")
    print("STEP 2: Generating Answer with Context")
    print(f"{'='*60}")
    print("Sending your question + retrieved documents (context) to the LLM...")
    print("The LLM will use this context to generate an answer.\n")
    
    answer = llm_client.generate_with_context(query, context_docs)
    
    # Display answer with metadata separated
    display_rag_answer(answer, metadata=metadata, retrieved_docs=results)
    
    return answer


def compare_all_search_methods(query: str, top_k: int = 3):
    """
    Compare keyword, semantic, vector, and hybrid search methods.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve for each method
    """
    print(f"\n{'='*60}")
    print("COMPREHENSIVE COMPARISON: All Search Methods")
    print(f"{'='*60}")
    print(f"\nQuery: {query}\n")
    
    search_client = AzureSearchClient()
    embedding_gen = EmbeddingGenerator()
    
    # Generate embedding for vector and hybrid search
    try:
        query_embedding = embedding_gen.generate_embedding(query)
    except Exception:
        query_embedding = None
        print("⚠ Could not generate embedding - skipping vector/hybrid search\n")
    
    methods = {
        "Keyword": lambda: search_client.search_keyword(query, top=top_k, use_pydantic=False),
        "Semantic": lambda: search_client.search_semantic(query, top=top_k),
        "Vector": lambda: search_client.search_vector(query_embedding, top=top_k) if query_embedding else [],
        "Hybrid": lambda: search_client.search_hybrid(query, query_embedding, top=top_k) if query_embedding else []
    }
    
    for method_name, search_func in methods.items():
        print(f"{method_name.upper()} SEARCH:")
        print("-" * 60)
        try:
            results = search_func()
            if not results:
                print(f"⚠ No results returned")
            else:
                for i, result in enumerate(results, 1):
                    content = result.get('content', result.get('text', ''))[:150]
                    score = result.get('@search.score', 'N/A')
                    reranker_score = result.get('@search.reranker_score', 'N/A')
                    score_str = f"Score: {score}"
                    if reranker_score != 'N/A':
                        score_str += f" | Semantic: {reranker_score}"
                    print(f"{i}. [{score_str}] {content}...")
        except Exception as e:
            print(f"⚠ {method_name} search not available: {str(e)[:100]}")
        print()
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Interactive mode
    print("\n" + "="*60)
    print("LAB 6: Semantic Search RAG - Interactive Mode")
    print("="*60)
    
    query = input("\nEnter your question: ").strip()
    if not query:
        query = "What is machine learning?"
        print(f"Using default query: {query}")
    
    # Run semantic RAG
    semantic_rag(query)
    
    # Compare all methods
    compare = input("\nCompare all search methods? (y/n): ").strip().lower()
    if compare == 'y':
        compare_all_search_methods(query)

