# Prompts Directory

This directory contains all prompts used in the RAG labs, consolidated in a single JSON file.

## Structure

All prompts are stored in `prompts.json` with a hierarchical structure:

```json
{
  "system": {
    "default": "...",
    "rag": "...",
    "rag_advanced": "..."
  },
  "user": {
    "rag": "..."
  },
  "multi_step": {
    "subqueries": "..."
  }
}
```

## Available Prompts

### System Prompts

- **system.default** - Default system prompt for general LLM interactions
- **system.rag** - System prompt for basic RAG operations
- **system.rag_advanced** - Enhanced system prompt for advanced RAG with citations

### User Prompts

- **user.rag** - Template for RAG user prompts with context and query
  - Variables: `{context_text}`, `{query}`

### Multi-Step Retrieval

- **multi_step.subqueries** - Prompt for generating sub-queries
  - Variables: `{query}`

## Usage

### Loading Prompts in Code

```python
from prompts.loader import load_prompt, load_template

# Load a simple prompt (use dot notation)
system_prompt = load_prompt("system.rag")

# Load and format a template
rag_prompt = load_template("user.rag", context_text=context, query=query)
```

### Adding New Prompts

1. Edit `prompts.json` and add your prompt to the appropriate section
2. Use `{variable_name}` for template variables
3. Load using `load_prompt()` or `load_template()` with dot notation

Example:
```json
{
  "system": {
    "custom": "Your custom system prompt here"
  }
}
```

Then load it:
```python
prompt = load_prompt("system.custom")
```

## Benefits

- **Single Source of Truth**: All prompts in one JSON file
- **Easy Editing**: Modify prompts without touching code
- **Version Control**: Track prompt changes independently
- **Reusability**: Share prompts across different labs
- **A/B Testing**: Easy to test different prompt variations
- **Structured Organization**: Hierarchical organization with dot notation

