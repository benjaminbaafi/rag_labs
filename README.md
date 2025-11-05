# RAG Labs - Educational Labs for AI Engineers

A comprehensive Python lab series teaching **Retrieval-Augmented Generation (RAG)** using **Azure AI Search** and **OpenAI**. Perfect for AI engineer interns learning RAG techniques!

## 🎯 Learning Objectives

This lab series covers:

1. **Basic RAG** - Simple retrieval and generation
2. **Vector-Based RAG** - Semantic search using embeddings
3. **Hybrid RAG** - Combining keyword and vector search
4. **Advanced RAG** - Multi-step retrieval, re-ranking, and context management
5. **Index Setup** - Configuring Azure AI Search for RAG

## 📋 Prerequisites

- Python 3.12+
- Azure subscription with:
  - Azure AI Search service
  - Azure OpenAI (optional) or OpenAI API access
- Basic Python knowledge

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Copy the example template
cp .env.example .env
```

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Edit `.env` with your actual credentials. You can use **either** Azure OpenAI **or** standard OpenAI:

**Option 1: Using Azure OpenAI (Recommended)**
```env
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-admin-key
AZURE_SEARCH_INDEX_NAME=rag-labs-index

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
OPENAI_API_VERSION=2024-02-15-preview
```

**Option 2: Using Standard OpenAI**
```env
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-admin-key
AZURE_SEARCH_INDEX_NAME=rag-labs-index

# Standard OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_DEPLOYMENT_NAME=gpt-4
```

**Note:** The system automatically detects which provider you've configured and uses it accordingly.

### 3. Setup Azure AI Search Index

First, run Lab 5 to create and populate your search index:

```bash
python main.py 5
```

Or run directly:

```bash
python labs/lab5_setup_index.py
```

### 4. Run the Labs

You can run labs in three ways:

#### Option 1: GUI Interface (Recommended)

For a user-friendly graphical interface:

```bash
uv run python main.py --gui
# or
uv run python gui.py
```

This opens a simple GUI window where you can:
- Select which lab to run
- Type your own questions
- See results in a text area
- Add custom text for Lab 1
- Run comparisons for Labs 2 & 3

#### Option 2: Interactive Terminal Menu

```bash
uv run python main.py
```

#### Option 3: Command Line

**Run Specific Lab:**
```bash
uv run python main.py 1  # Lab 1: Basic RAG
uv run python main.py 2  # Lab 2: Vector-Based RAG
uv run python main.py 3  # Lab 3: Hybrid RAG
uv run python main.py 4  # Lab 4: Advanced RAG
uv run python main.py 5  # Lab 5: Setup Index
```

**Run Labs Directly:**
```bash
python labs/lab1_basic_rag.py
python labs/lab2_vector_rag.py
python labs/lab3_hybrid_rag.py
python labs/lab4_advanced_rag.py
```

## 📚 Lab Details

### Lab 1: Basic RAG

**What you'll learn:**
- Understanding the RAG pipeline
- Retrieving documents from Azure AI Search
- Injecting context into LLM prompts

**Key Concepts:**
- Keyword search
- Context injection
- Prompt engineering basics

```python
from labs import lab1_basic_rag
lab1_basic_rag.basic_rag("What is machine learning?")
```

### Lab 2: Vector-Based RAG

**What you'll learn:**
- Generating embeddings for queries
- Vector similarity search
- Semantic understanding vs. keyword matching

**Key Concepts:**
- Embeddings
- Vector search
- Semantic similarity

```python
from labs import lab2_vector_rag
lab2_vector_rag.vector_rag("How does neural network training work?")
```

### Lab 3: Hybrid RAG

**What you'll learn:**
- Combining keyword and vector search
- When to use hybrid vs. single method
- Benefits of hybrid search

**Key Concepts:**
- Hybrid search
- Search result fusion
- Balanced retrieval strategies

```python
from labs import lab3_hybrid_rag
lab3_hybrid_rag.hybrid_rag("Explain deep learning architectures")
```

### Lab 4: Advanced RAG

**What you'll learn:**
- Multi-step retrieval (retrieve, refine, retrieve again)
- Context window management
- Advanced prompt engineering
- Re-ranking strategies

**Key Concepts:**
- Iterative retrieval
- Token counting and management
- Enhanced prompts for RAG

```python
from labs import lab4_advanced_rag
lab4_advanced_rag.advanced_rag(
    "What are the latest advances in transformer models?",
    use_multi_step=True,
    use_context_management=True
)
```

### Lab 5: Setup Azure AI Search Index

**What you'll learn:**
- Creating Azure AI Search indexes
- Configuring vector search
- Setting up semantic search
- Uploading documents with embeddings

**Key Concepts:**
- Index schema design
- Vector field configuration
- Document ingestion

## 🏗️ Project Structure

```
rag_labs/
├── labs/                    # Lab exercises
│   ├── lab1_basic_rag.py
│   ├── lab2_vector_rag.py
│   ├── lab3_hybrid_rag.py
│   ├── lab4_advanced_rag.py
│   └── lab5_setup_index.py
├── utils/                   # Utility modules
│   ├── azure_search.py     # Azure AI Search client
│   ├── embeddings.py       # Embedding generation
│   └── llm.py              # LLM client
├── config.py               # Configuration management
├── main.py                 # Main entry point
├── pyproject.toml          # Dependencies
└── README.md              # This file
```

## 🔧 Key Components

### AzureSearchClient (`utils/azure_search.py`)
- Keyword search
- Vector search
- Hybrid search
- Semantic search

### EmbeddingGenerator (`utils/embeddings.py`)
- Generate embeddings for queries
- Batch embedding generation
- Supports OpenAI and Azure OpenAI

### LLMClient (`utils/llm.py`)
- Generate responses with context
- Customizable system prompts
- Supports OpenAI and Azure OpenAI

## 💡 Tips for Interns

1. **Start with Lab 1** - Understand the basics before moving to advanced techniques
2. **Experiment with queries** - Try different questions to see how retrieval changes
3. **Compare methods** - Use the comparison functions to see differences between search types
4. **Read the code** - Each lab has detailed comments explaining concepts
5. **Modify parameters** - Change `top_k`, `temperature`, etc. to see the impact

## 🐛 Troubleshooting

**"No documents found"**
- Run Lab 5 first to create and populate the index
- Check that your index name matches in `.env`

**"Missing required environment variables"**
- Ensure your `.env` file exists and has all required variables
- Check that variable names match exactly

**"Authentication failed"**
- Verify your Azure Search endpoint and key are correct
- Check that your OpenAI API key is valid

**Import errors**
- Run `uv sync` or `pip install -r requirements.txt` to install dependencies

## 📖 Additional Resources

- [Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [RAG Papers and Research](https://arxiv.org/abs/2005.11401)

## 🤝 Contributing

This is an educational lab. Feel free to:
- Add more lab examples
- Improve documentation
- Fix bugs
- Share your solutions!

## 📝 License

Educational use - feel free to modify and use for teaching purposes.

---

**Happy Learning! 🚀**

