"""
Lab 1: Basic RAG (Retrieval-Augmented Generation)
==================================================

This lab demonstrates the simplest form of RAG:
1. Add your own text to the search index
2. Retrieve relevant documents from Azure AI Search
3. Pass retrieved documents as context to LLM
4. Generate answer based on context

Learning Objectives:
- Understand the basic RAG pipeline
- Learn how to add documents to Azure AI Search
- Learn how to retrieve documents from Azure AI Search
- Learn how to inject context into LLM prompts
"""

from utils import AzureSearchClient, LLMClient, EmbeddingGenerator
from utils.display import display_rag_answer, extract_metadata_from_results


def add_text_to_index(text: str, doc_id: str = "lab1-doc", title: str = "Lab 1 Document"):
    """
    Add a text document to the search index with embeddings.
    
    Args:
        text: Text content to add
        doc_id: Document ID
        title: Document title
    """
    print("Adding text to search index...")
    search_client = AzureSearchClient()
    embedding_gen = EmbeddingGenerator()
    
    # Generate embedding for the text
    print("Generating embedding...")
    embedding = embedding_gen.generate_embedding(text)
    
    # Create document - try to match common index schemas
    # Adjust these fields based on your actual index schema
    document = {
        "id": doc_id,
    }
    
    # Add text content - try common field names
    # Check your index schema and use the correct field name for text content
    # Common field names: "content", "text", "body", "description"
    document["content"] = text  # or "text" or whatever your index uses
    
    # Add vector if your index supports it
    # Check if your index has a vector field (e.g., "content_vector", "contentVector", "embedding", "vector")
    # Note: We'll validate the schema when uploading, so we just add it here
    document["content_vector"] = embedding  # or "contentVector" or "embedding" or "vector"
    
    # Upload to index
    try:
        result = search_client.upload_documents([document])
        print(f"✓ Document added to index (ID: {doc_id})\n")
    except Exception:
        print(f"⚠ Could not add document with vector. Trying without vector...")
        # Retry without vector field (for keyword-only search)
        document_no_vector = {"id": doc_id, "content": text}
        try:
            result = search_client.upload_documents([document_no_vector])
            print(f"✓ Document added to index (keyword search only, ID: {doc_id})\n")
        except Exception as e2:
            print(f"✗ Error adding document: {e2}")
            print("Please check your index schema and field names.")
            raise
    
    return document


def basic_rag(query: str, top_k: int = 3, use_custom_text: bool = False, custom_text: str = None):
    """
    Perform basic RAG: retrieve documents and generate answer.
    
    RAG Flow (when custom text is provided):
    1. Use custom text directly as context (no index search needed)
    2. Send the custom text + question to the LLM
    3. LLM generates an answer based on the custom text
    
    RAG Flow (when no custom text):
    1. Search the Azure AI Search index
    2. Retrieve top-k most relevant documents (these become the "context")
    3. Send the context + question to the LLM
    4. LLM generates an answer based on the context
    
    The "context" is the text that provides information to answer your question.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve (only used if no custom text)
        use_custom_text: If True, use custom text directly as context (no index search)
        custom_text: Custom text to use as context (if use_custom_text is True)
    """
    print(f"\n{'='*60}")
    print("LAB 1: Basic RAG")
    print(f"{'='*60}")
    
    print(f"\nQuery: {query}\n")
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    # If custom text is provided, use it directly (no index search)
    if use_custom_text and custom_text:
        print("-"*60)
        print("STEP 1: Using Custom Text as Context")
        print("-"*60)
        print("Using your custom text directly as context (no index search).")
        print("This demonstrates RAG with your specific text.\n")
        
        # Create a simple document from custom text
        from models.search import Document
        custom_doc = Document(
            id="custom-text",
            content=custom_text.strip(),
            title="Custom Text"
        )
        
        print("Custom Text (Context):")
        print("-"*60)
        print(custom_text.strip()[:500] + ("..." if len(custom_text) > 500 else ""))
        print("-"*60)
        print("\nThis text will be used as context for the LLM.\n")
        
        # Step 2: Generate answer with custom text as context
        print(f"{'='*60}")
        print("STEP 2: Generating Answer with Custom Text Context")
        print(f"{'='*60}")
        print("Sending your question + custom text (context) to the LLM...")
        print("The LLM will use this context to generate an answer.\n")
        
        answer = llm_client.generate_with_context(query, [custom_doc])
        
        # Display answer with minimal metadata
        metadata = {
            'num_documents': 1,
            'search_type': 'custom_text',
            'scores': []
        }
        display_rag_answer(answer, metadata=metadata, retrieved_docs=[custom_doc])
        
    else:
        # No custom text - search the index
        print("-"*60)
        print("STEP 1: Retrieving Documents from Azure AI Search")
        print("-"*60)
        print("Searching the Azure AI Search index for relevant documents...\n")
        
        # Initialize search client
        search_client = AzureSearchClient()
        
        # Retrieve relevant documents
        search_response = search_client.search_keyword(query, top=top_k, use_pydantic=True)
        
        if not search_response or not search_response.results:
            print("No documents found. Please ensure your search index is populated.")
            print("Tip: Use custom text option to demonstrate RAG without an index.")
            return
        
        # Extract metadata
        metadata = extract_metadata_from_results(search_response.results, search_type="keyword")
        metadata['num_documents'] = len(search_response.results)
        
        # Extract scores
        scores = [result.score for result in search_response.results]
        metadata['scores'] = scores
        
        # Get context documents
        context_docs = [result.document for result in search_response.results]
        
        # Step 2: Generate answer with context
        print(f"\n{'='*60}")
        print("STEP 2: Generating Answer with Context")
        print(f"{'='*60}")
        print("Sending your question + the retrieved documents (context) to the LLM...")
        print("The LLM will use this context to generate an answer.\n")
        
        answer = llm_client.generate_with_context(query, context_docs)
        
        # Display answer with metadata separated
        display_rag_answer(answer, metadata=metadata, retrieved_docs=search_response.results)
    
    return answer


if __name__ == "__main__":
    # Interactive mode - get user input
    print("\n" + "="*60)
    print("LAB 1: Basic RAG - Interactive Mode")
    print("="*60)
    
    # Get query
    query = input("\nEnter your question: ").strip()
    if not query:
        query = "What is machine learning?"
        print(f"Using default query: {query}")
    
    # Ask about custom text
    use_custom = input("\nDo you want to add custom text? (y/n): ").strip().lower()
    custom_text = None
    if use_custom == 'y':
        print("\nEnter your custom text (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        custom_text = "\n".join(lines[:-1])  # Remove last empty line
        
        if not custom_text.strip():
            print("No text entered, using default...")
            custom_text = """
            Machine learning is a subset of artificial intelligence that enables computers to learn 
            and make decisions from data without being explicitly programmed.
            """
    
    # Run RAG
    basic_rag(query, use_custom_text=use_custom == 'y', custom_text=custom_text)

