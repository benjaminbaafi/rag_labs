"""
Lab 4: Advanced RAG with Re-ranking and Multi-step Retrieval
============================================================

This lab demonstrates advanced RAG techniques:
1. Multi-step retrieval (retrieve, refine, retrieve again)
2. Re-ranking results
3. Context window management
4. Prompt engineering for better results

Learning Objectives:
- Understand iterative retrieval strategies
- Learn context window management techniques
- Explore prompt engineering for RAG
"""

from utils import AzureSearchClient, LLMClient, EmbeddingGenerator
from utils.display import display_rag_answer, extract_metadata_from_results
from typing import List, Dict, Any
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def manage_context_window(
    documents: List[Dict[str, Any]],
    max_tokens: int = 2000,
    model: str = "gpt-4"
) -> List[Dict[str, Any]]:
    """
    Manage context window by selecting documents that fit within token limit.
    
    Args:
        documents: List of retrieved documents
        max_tokens: Maximum tokens allowed
        model: Model name for token counting
        
    Returns:
        Filtered list of documents that fit within token limit
    """
    selected_docs = []
    current_tokens = 0
    
    for doc in documents:
        # Handle both dict and Pydantic models
        if isinstance(doc, dict):
            content = doc.get('content', doc.get('text', ''))
        elif hasattr(doc, 'content'):
            content = doc.content
        elif hasattr(doc, 'document'):
            content = doc.document.content
        else:
            content = str(doc)
        
        doc_tokens = count_tokens(content, model)
        
        if current_tokens + doc_tokens <= max_tokens:
            selected_docs.append(doc)
            current_tokens += doc_tokens
        else:
            # Try to truncate document if it's the first one
            if not selected_docs:
                # Truncate to fit
                encoding = tiktoken.encoding_for_model(model)
                tokens = encoding.encode(content)
                truncated_tokens = tokens[:max_tokens]
                truncated_content = encoding.decode(truncated_tokens)
                doc_copy = doc.copy()
                doc_copy['content'] = truncated_content
                selected_docs.append(doc_copy)
            break
    
    return selected_docs


def multi_step_retrieval(
    query: str,
    search_client: AzureSearchClient,
    embedding_gen: EmbeddingGenerator,
    llm_client: LLMClient,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform multi-step retrieval: retrieve, generate sub-query, retrieve again.
    
    Args:
        query: Initial query
        search_client: Azure Search client
        embedding_gen: Embedding generator
        llm_client: LLM client
        top_k: Number of documents per retrieval step
        
    Returns:
        Combined list of retrieved documents
    """
    print("Step 1: Initial retrieval...")
    initial_results_raw = search_client.search_keyword(query, top=top_k, use_pydantic=False)
    initial_results = initial_results_raw if isinstance(initial_results_raw, list) else initial_results_raw.results if hasattr(initial_results_raw, 'results') else []
    print(f"Retrieved {len(initial_results)} documents\n")
    
    # Generate sub-queries to refine search
    print("Step 2: Generating sub-queries for refined retrieval...")
    from prompts.loader import load_template
    try:
        sub_query_prompt = load_template("multi_step.subqueries", query=query)
    except FileNotFoundError:
        sub_query_prompt = f"""Based on the initial query and retrieved documents, generate 2-3 more specific sub-queries that would help find additional relevant information.

Initial query: {query}

Generate sub-queries as a numbered list:"""
    
    sub_queries_text = llm_client.generate(sub_query_prompt, max_tokens=200)
    print(f"Generated sub-queries:\n{sub_queries_text}\n")
    
    # Extract sub-queries (simple extraction - could be improved)
    lines = sub_queries_text.split('\n')
    sub_queries = [
        line.split('. ', 1)[1] if '. ' in line else line.strip()
        for line in lines
        if line.strip() and any(char.isdigit() for char in line[:3])
    ][:2]  # Take first 2 sub-queries
    
    # Retrieve using sub-queries
    all_results = initial_results.copy()
    seen_ids = {r.get('id', '') for r in initial_results}
    
    for sub_query in sub_queries:
        print(f"Retrieving with sub-query: {sub_query}")
        sub_results_raw = search_client.search_keyword(sub_query, top=top_k, use_pydantic=False)
        sub_results = sub_results_raw if isinstance(sub_results_raw, list) else sub_results_raw.results if hasattr(sub_results_raw, 'results') else []
        for result in sub_results:
            result_id = result.get('id', '') if isinstance(result, dict) else (result.document.id if hasattr(result, 'document') else result.id if hasattr(result, 'id') else '')
            if result_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result_id)
    
    print(f"\nTotal unique documents retrieved: {len(all_results)}\n")
    return all_results


def advanced_rag(
    query: str,
    use_multi_step: bool = True,
    use_context_management: bool = True,
    top_k: int = 5
):
    """
    Perform advanced RAG with multiple techniques.
    
    Args:
        query: User question
        use_multi_step: Use multi-step retrieval
        use_context_management: Use context window management
        top_k: Number of documents to retrieve
    """
    print(f"\n{'='*60}")
    print("LAB 4: Advanced RAG")
    print(f"{'='*60}")
    print(f"\nQuery: {query}\n")
    
    # Initialize clients
    search_client = AzureSearchClient()
    llm_client = LLMClient()
    embedding_gen = EmbeddingGenerator()
    
    # Step 1: Retrieve documents
    if use_multi_step:
        results = multi_step_retrieval(query, search_client, embedding_gen, llm_client, top_k)
    else:
        results = search_client.search_keyword(query, top=top_k)
    
    if not results:
        print("No documents found.")
        return
    
    # Step 2: Manage context window
    if use_context_management:
        print("Step 3: Managing context window...")
        max_context_tokens = 2000
        results = manage_context_window(results, max_tokens=max_context_tokens)
        print(f"Selected {len(results)} documents within token limit\n")
    
    # Extract metadata
    metadata = extract_metadata_from_results(results, search_type="advanced")
    if use_multi_step:
        metadata['additional_info'] = {'multi_step': True}
    if use_context_management:
        metadata['additional_info'] = metadata.get('additional_info', {})
        metadata['additional_info']['context_management'] = True
    
    # Step 3: Generate answer with enhanced prompt
    print("Step 4: Generating answer with enhanced prompt...")
    
    from prompts.loader import load_prompt
    try:
        enhanced_system_prompt = load_prompt("system.rag_advanced")
    except FileNotFoundError:
        enhanced_system_prompt = """You are an expert assistant that answers questions based on provided context.
- Use only information from the provided context
- If the context doesn't contain enough information, say so
- Cite specific documents when making claims
- Be concise but thorough"""
    
    answer = llm_client.generate_with_context(
        query,
        results,
        system_prompt=enhanced_system_prompt
    )
    
    # Display answer with metadata separated
    display_rag_answer(answer, metadata=metadata, retrieved_docs=results)
    
    return answer


if __name__ == "__main__":
    # Interactive mode
    print("\n" + "="*60)
    print("LAB 4: Advanced RAG - Interactive Mode")
    print("="*60)
    
    query = input("\nEnter your question: ").strip()
    if not query:
        query = "What are the latest advances in transformer models?"
        print(f"Using default query: {query}")
    
    advanced_rag(query, use_multi_step=True, use_context_management=True)

