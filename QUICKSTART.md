# Quick Start Guide for AI Engineer Interns

Welcome to the RAG Labs! This guide will help you get started quickly.

## 🎯 What You'll Learn

This lab series teaches you **Retrieval-Augmented Generation (RAG)** - a powerful technique that combines:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Creating natural language responses using AI

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies

```bash
# If using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 2: Get Your Credentials

You'll need:
1. **Azure AI Search** - Get your endpoint and key from Azure Portal
2. **Either Azure OpenAI OR Standard OpenAI:**
   - **Azure OpenAI** - Get your endpoint, API key, and deployment names from Azure Portal
   - **Standard OpenAI** - Get your API key from https://platform.openai.com/api-keys

### Step 3: Create .env File

Copy the example template and fill in your credentials:

```bash
# On Linux/Mac
cp .env.example .env

# On Windows (PowerShell)
Copy-Item .env.example .env
```

Then edit `.env` with your actual values. You can use **either** Azure OpenAI **or** standard OpenAI:

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

**Note:** The `.env` file is already in `.gitignore` - never commit your actual credentials!

### Step 4: Setup Your Index

Run Lab 5 to create and populate your search index:

```bash
python main.py 5
```

This will:
- Create an Azure AI Search index
- Upload sample documents with embeddings
- Configure the index for RAG

### Step 5: Run Your First Lab!

```bash
python main.py 1
```

## 📚 Lab Progression

Follow this order for best learning:

1. **Lab 1: Basic RAG** ← Start here!
   - Simple retrieval + generation
   - Understand the RAG pipeline

2. **Lab 2: Vector-Based RAG**
   - Semantic search with embeddings
   - Better understanding of context

3. **Lab 3: Hybrid RAG**
   - Combine keyword + vector search
   - Best of both worlds

4. **Lab 4: Advanced RAG**
   - Multi-step retrieval
   - Context management
   - Advanced techniques

5. **Lab 5: Setup Index**
   - Learn to configure Azure AI Search
   - Understand index structure

## 💡 Tips for Success

1. **Read the code comments** - Each lab has detailed explanations
2. **Experiment** - Change queries and parameters to see what happens
3. **Compare methods** - Use comparison functions to see differences
4. **Ask questions** - Understanding is more important than speed

## 🔍 Understanding RAG

**RAG = Retrieval + Augmented Generation**

```
User Query
    ↓
[Retrieve] → Find relevant documents from knowledge base
    ↓
[Augment] → Add retrieved documents to LLM prompt
    ↓
[Generate] → LLM creates answer using context
    ↓
Answer
```

## 🎓 Learning Path

### Beginner
- Lab 1: Understand basic RAG
- Lab 2: Learn about embeddings

### Intermediate
- Lab 3: Explore hybrid search
- Modify labs to experiment

### Advanced
- Lab 4: Advanced techniques
- Lab 5: Infrastructure setup

## 🐛 Common Issues

**"No documents found"**
→ Run Lab 5 first to create the index

**"Missing environment variables"**
→ Check your `.env` file exists and has all variables

**"Import errors"**
→ Run `uv sync` or `pip install -r requirements.txt`

## 📖 Next Steps

After completing the labs:
1. Try your own queries
2. Modify the code to experiment
3. Add your own documents to the index
4. Explore advanced RAG techniques

## 🚀 Have Fun Learning!

Remember: The best way to learn is by doing. Don't be afraid to experiment and break things - that's how you learn!

---

**Questions?** Check the main README.md for detailed documentation.

