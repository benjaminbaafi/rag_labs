"""Test script to verify RAG labs setup."""
import sys

print("=" * 60)
print("RAG Labs Setup Verification")
print("=" * 60)
print()

# Test 1: Configuration
print("1. Testing Configuration...")
try:
    import config
    cfg = config.get_config()
    cfg.validate()
    print("   ✓ Configuration loaded and validated")
    
    # Show which provider is configured
    if cfg.uses_azure_openai:
        print(f"   ✓ Using Azure OpenAI")
        print(f"     - Endpoint: {cfg.azure_openai_endpoint}")
        print(f"     - Deployment: {cfg.azure_openai_deployment_name}")
        print(f"     - Embedding: {cfg.azure_openai_embedding_deployment}")
    elif cfg.uses_openai:
        print(f"   ✓ Using Standard OpenAI")
        print(f"     - Deployment: {cfg.openai_deployment_name}")
    else:
        print("   ✗ No OpenAI provider configured")
        sys.exit(1)
    
    print(f"   ✓ Azure AI Search configured")
    print(f"     - Endpoint: {cfg.search_endpoint}")
    print(f"     - Index: {cfg.azure_search_index_name}")
except Exception as e:
    print(f"   ✗ Configuration error: {e}")
    sys.exit(1)

print()

# Test 2: Azure Search Client
print("2. Testing Azure AI Search Client...")
try:
    from utils import AzureSearchClient
    search_client = AzureSearchClient()
    print("   ✓ Azure Search client initialized")
except Exception as e:
    print(f"   ✗ Azure Search client error: {e}")
    sys.exit(1)

print()

# Test 3: LLM Client
print("3. Testing LLM Client...")
try:
    from utils import LLMClient
    llm_client = LLMClient()
    print(f"   ✓ LLM client initialized")
    print(f"     - Provider: {llm_client.provider}")
    print(f"     - Deployment: {llm_client.deployment}")
except Exception as e:
    print(f"   ✗ LLM client error: {e}")
    sys.exit(1)

print()

# Test 4: Embedding Generator
print("4. Testing Embedding Generator...")
try:
    from utils import EmbeddingGenerator
    embedding_gen = EmbeddingGenerator()
    print(f"   ✓ Embedding generator initialized")
    print(f"     - Provider: {embedding_gen.provider}")
    print(f"     - Deployment: {embedding_gen.deployment}")
except Exception as e:
    print(f"   ✗ Embedding generator error: {e}")
    sys.exit(1)

print()

# Test 5: Optional - Test actual API call (commented out by default)
print("5. Optional API Tests (skipped by default)")
print("   To test actual API calls, uncomment the code below")
print("   and run: python test_setup.py --test-api")
print()

if "--test-api" in sys.argv:
    print("   Testing Embedding Generation...")
    try:
        test_text = "This is a test"
        embedding = embedding_gen.generate_embedding(test_text)
        print(f"   ✓ Embedding generated: {len(embedding)} dimensions")
    except Exception as e:
        print(f"   ✗ Embedding generation error: {e}")
    
    print()
    print("   Testing LLM Generation...")
    try:
        response = llm_client.generate(
            "Say 'Hello, RAG Labs!' in one sentence.",
            max_tokens=50
        )
        print(f"   ✓ LLM response received: {response[:100]}...")
    except Exception as e:
        print(f"   ✗ LLM generation error: {e}")

print()
print("=" * 60)
print("Setup Verification Complete!")
print("=" * 60)
print()
print("All checks passed! ✓ Your RAG labs are ready to use.")
print()
print("Next steps:")
print("  1. Run Lab 5 to set up your search index: python main.py 5")
print("  2. Then run your first lab: python main.py 1")
print()

