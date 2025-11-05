# Lab 1: Understanding RAG and Context

## What is "Context" in RAG?

**Context** = The retrieved documents that provide information to answer your question.

## How Lab 1 Works with Custom Text

### Step-by-Step Flow:

1. **Add Custom Text (if provided)**
   - Your custom text is added to the Azure AI Search index
   - The index now contains: **Your custom text + All existing documents**
   - The custom text becomes searchable like any other document

2. **Search the Index**
   - When you ask a question, we search the **entire index**
   - This includes your custom text AND all existing documents
   - Azure AI Search finds the most relevant documents

3. **Retrieve Top Documents**
   - The top-k most relevant documents are retrieved
   - These retrieved documents become the **"context"**
   - If your custom text is relevant, it might be in the top results

4. **Send to LLM**
   - Your question + the retrieved documents (context) are sent to the LLM
   - The LLM uses this context to generate an answer
   - The LLM can only use information from the context

## Example Flow:

```
Question: "What is machine learning?"

Step 1: Add custom text about ML to index
Step 2: Search entire index (custom text + existing docs)
Step 3: Retrieve top 3 most relevant documents → These are the "context"
Step 4: Send to LLM:
   - Question: "What is machine learning?"
   - Context: [Document 1, Document 2, Document 3]
Step 5: LLM generates answer based on the context
```

## Why Use the Index?

The index acts as a **knowledge base**:
- It stores all searchable documents
- It allows fast retrieval of relevant information
- It can contain thousands of documents
- Your custom text becomes part of this knowledge base

## Key Points:

1. **Custom text is ADDED to the index, not replaced**
   - The index contains your custom text + existing documents
   - Both are searchable

2. **Context = Retrieved Documents**
   - The "context" is the documents retrieved from the index
   - These documents provide information to answer your question

3. **LLM uses Context to Answer**
   - The LLM doesn't search the index directly
   - It uses the retrieved documents (context) to generate answers

4. **Relevance Matters**
   - If your custom text is relevant to your question, it might be retrieved
   - If other documents are more relevant, they'll be retrieved instead
   - This is why you might not see your custom text in the results

## Visual Representation:

```
┌─────────────────────────────────────────┐
│  Azure AI Search Index                  │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ Your Custom │  │ Existing    │      │
│  │ Text        │  │ Documents    │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
           ↓
    Search Query
           ↓
┌─────────────────────────────────────────┐
│  Retrieved Documents (Context)          │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ Document 1  │  │ Document 2  │      │
│  │ (Score: 0.9)│  │ (Score: 0.8)│      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
           ↓
    Question + Context
           ↓
┌─────────────────────────────────────────┐
│  LLM (OpenAI / Azure OpenAI)            │
│  Generates answer based on context      │
└─────────────────────────────────────────┘
           ↓
    Final Answer
```

## Common Questions:

**Q: Why is my custom text not in the results?**
A: Other documents in the index were more relevant to your question. Try:
- Making your question match your custom text better
- Increasing `top_k` to see more results
- Ensuring your custom text is actually relevant to your question

**Q: Does the LLM search the index?**
A: No! The LLM only sees the retrieved documents (context). It doesn't search the index itself.

**Q: What if I want to use ONLY my custom text?**
A: You would need to:
- Create a separate index with only your custom text, OR
- Filter the search to only return documents matching your custom text ID

**Q: Can I see what context is sent to the LLM?**
A: Yes! The retrieved documents are displayed before the answer. These are the documents used as context.

