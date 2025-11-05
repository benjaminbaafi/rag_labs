"""
Display utilities for RAG labs.
Separates answer text from metadata for cleaner output.
"""
from typing import List, Dict, Any, Optional


def display_rag_answer(
    answer: str,
    metadata: Optional[Dict[str, Any]] = None,
    retrieved_docs: Optional[List] = None
):
    """
    Display RAG answer with metadata separated from answer text.
    
    Args:
        answer: The generated answer text (clean, no metadata)
        metadata: Optional metadata dictionary with keys like:
            - num_documents: Number of documents retrieved
            - search_type: Type of search performed
            - scores: List of relevance scores
            - retrieval_time: Time taken for retrieval
            - generation_time: Time taken for generation
        retrieved_docs: Optional list of retrieved documents (for display)
    """
    # Display metadata section (if provided)
    if metadata or retrieved_docs:
        print(f"\n{'='*60}")
        print("METADATA")
        print(f"{'='*60}")
        
        if metadata:
            if metadata.get('num_documents'):
                print(f"Documents Retrieved: {metadata['num_documents']}")
            if metadata.get('search_type'):
                print(f"Search Type: {metadata['search_type']}")
            if metadata.get('scores'):
                scores = metadata['scores']
                if scores:
                    avg_score = sum(scores) / len(scores) if scores else 0
                    print(f"Average Relevance Score: {avg_score:.4f}")
                    print(f"Score Range: {min(scores):.4f} - {max(scores):.4f}")
            if metadata.get('retrieval_time'):
                print(f"Retrieval Time: {metadata['retrieval_time']:.2f}s")
            if metadata.get('generation_time'):
                print(f"Generation Time: {metadata['generation_time']:.2f}s")
            if metadata.get('total_tokens'):
                print(f"Total Tokens: {metadata['total_tokens']}")
        
        # Display retrieved documents summary (if provided)
        if retrieved_docs:
            print(f"\nRetrieved Documents Summary:")
            print("-" * 60)
            for i, doc in enumerate(retrieved_docs[:5], 1):  # Show first 5
                # Handle both dict and Document models
                if isinstance(doc, dict):
                    content = doc.get('content', doc.get('text', ''))
                    doc_id = doc.get('id', f'Doc {i}')
                elif hasattr(doc, 'document'):
                    # SearchResult model
                    content = doc.document.content
                    doc_id = doc.document.id if hasattr(doc.document, 'id') else f'Doc {i}'
                    score = getattr(doc, 'score', None)
                    if score:
                        print(f"  Document {i} (ID: {doc_id}, Score: {score:.4f}): {content[:100]}...")
                    else:
                        print(f"  Document {i} (ID: {doc_id}): {content[:100]}...")
                    continue
                elif hasattr(doc, 'content'):
                    # Document model
                    content = doc.content
                    doc_id = doc.id if hasattr(doc, 'id') else f'Doc {i}'
                else:
                    content = str(doc)
                    doc_id = f'Doc {i}'
                
                print(f"  Document {i} (ID: {doc_id}): {content[:100]}...")
            
            if len(retrieved_docs) > 5:
                print(f"  ... and {len(retrieved_docs) - 5} more document(s)")
    
    # Display answer section (clean, just the answer)
    print(f"\n{'='*60}")
    print("ANSWER")
    print(f"{'='*60}")
    print(answer)
    print(f"{'='*60}\n")


def extract_metadata_from_results(
    results: List,
    search_type: str = "unknown"
) -> Dict[str, Any]:
    """
    Extract metadata from search results.
    
    Args:
        results: List of search results (dict or SearchResult models)
        search_type: Type of search performed
        
    Returns:
        Dictionary with metadata
    """
    metadata = {
        'num_documents': 0,
        'search_type': search_type,
        'scores': []
    }
    
    if not results:
        return metadata
    
    # Handle SearchResponse model
    if hasattr(results, 'results'):
        results_list = results.results
    else:
        results_list = results
    
    metadata['num_documents'] = len(results_list)
    
    # Extract scores
    scores = []
    for result in results_list:
        if isinstance(result, dict):
            score = result.get('@search.score', result.get('score'))
            if score:
                scores.append(float(score))
        elif hasattr(result, 'score'):
            scores.append(float(result.score))
        elif hasattr(result, 'document') and hasattr(result.document, 'score'):
            scores.append(float(result.document.score))
    
    if scores:
        metadata['scores'] = scores
    
    return metadata

