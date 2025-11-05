"""
AI Agent Framework
=================
An agent that can reason about which tool to use and execute actions.
"""

import json
import re
from typing import List, Dict, Any, Optional
from utils.llm import LLMClient
from utils.sql_tool import SQLTool
from utils.azure_search import AzureSearchClient
from utils.display import display_rag_answer, extract_metadata_from_results
from prompts.loader import load_prompt, load_template


class AIAgent:
    """
    AI Agent that can use multiple tools (RAG, SQL) intelligently.
    
    The agent reasons about which tool to use based on the query.
    """
    
    def __init__(self):
        """Initialize the AI agent with available tools."""
        self.llm_client = LLMClient()
        self.sql_tool = SQLTool()
        self.rag_client = AzureSearchClient()
        
        # Tool descriptions for the agent
        self.tools = {
            "rag": {
                "name": "RAG (Retrieval-Augmented Generation)",
                "description": self._get_rag_tool_description(),
                "use_when": "User asks conceptual questions, needs explanations, wants to understand topics, or asks 'what is', 'explain', 'how does', 'tell me about'"
            },
            "sql": {
                "name": "SQL Database",
                "description": self.sql_tool.describe_tool(),
                "use_when": "User asks for specific facts, numbers, statistics, lists, or structured data queries"
            }
        }
    
    def _get_rag_tool_description(self) -> str:
        """Get RAG tool description."""
        return """
RAG Tool - Retrieve and generate answers from knowledge base.

Use this tool when:
- User asks conceptual questions ("What is machine learning?")
- User wants explanations or understanding
- User asks "explain", "how does", "tell me about"
- User needs information from documents/knowledge base
- Query is fuzzy or requires semantic understanding
"""
    
    def reason_and_execute(self, query: str) -> Dict[str, Any]:
        """
        Reason about which tool to use and execute the query.
        
        Args:
            query: User query/question
            
        Returns:
            Dictionary with tool_used, answer, and metadata
        """
        # Step 1: Reason about which tool to use
        tool_decision = self._decide_tool(query)
        tool_name = tool_decision["tool"]
        reasoning = tool_decision["reasoning"]
        
        print(f"\n{'='*60}")
        print("AGENT REASONING")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"\nReasoning: {reasoning}")
        print(f"Selected Tool: {tool_name.upper()}")
        print(f"{'='*60}\n")
        
        # Step 2: Execute using selected tool
        if tool_name == "sql":
            result = self._execute_sql(query)
        elif tool_name == "rag":
            result = self._execute_rag(query)
        else:
            result = {
                "success": False,
                "answer": f"Unknown tool: {tool_name}",
                "tool_used": tool_name,
                "metadata": {}
            }
        
        return result
    
    def _decide_tool(self, query: str) -> Dict[str, Any]:
        """
        Decide which tool to use based on the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with tool name and reasoning
        """
        # Load tool selection prompt
        try:
            prompt = load_template(
                "agent.tool_selection",
                query=query,
                rag_description=self.tools["rag"]["description"],
                sql_description=self.tools["sql"]["description"],
                rag_use_when=self.tools["rag"]["use_when"],
                sql_use_when=self.tools["sql"]["use_when"]
            )
        except FileNotFoundError:
            prompt = f"""You are an AI agent that needs to decide which tool to use for a user query.

Available Tools:
1. RAG (Retrieval-Augmented Generation): For conceptual questions, explanations, understanding topics
   Use when: {self.tools['rag']['use_when']}
   
2. SQL Database: For specific facts, numbers, statistics, structured data queries
   Use when: {self.tools['sql']['use_when']}

User Query: {query}

Analyze the query and determine which tool is most appropriate.
Respond in JSON format:
{{
    "tool": "rag" or "sql",
    "reasoning": "Brief explanation of why this tool was chosen"
}}
"""
        
        try:
            system_prompt = load_prompt("system.agent")
        except FileNotFoundError:
            system_prompt = "You are an intelligent AI agent that reasons about which tool to use for user queries. Respond only with valid JSON."
        
        response = self.llm_client.generate(prompt, system_prompt=system_prompt, temperature=0.3, max_tokens=200)
        
        # Parse JSON response
        try:
            # Extract JSON from response (might have extra text)
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                # Fallback: simple heuristic
                decision = self._fallback_tool_decision(query)
        except Exception:
            # Fallback: use simple heuristic
            decision = self._fallback_tool_decision(query)
        
        return decision
    
    def _fallback_tool_decision(self, query: str) -> Dict[str, Any]:
        """
        Fallback tool decision using simple heuristics.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with tool name and reasoning
        """
        query_lower = query.lower()
        
        # SQL indicators
        sql_keywords = ["how many", "count", "list all", "find", "total", "average", "sum", "what is the", "which", "who has"]
        sql_indicators = any(keyword in query_lower for keyword in sql_keywords)
        
        # RAG indicators
        rag_keywords = ["what is", "explain", "how does", "tell me about", "describe", "understand", "why", "concept"]
        rag_indicators = any(keyword in query_lower for keyword in rag_keywords)
        
        if sql_indicators and not rag_indicators:
            return {
                "tool": "sql",
                "reasoning": "Query appears to request specific facts or structured data"
            }
        else:
            return {
                "tool": "rag",
                "reasoning": "Query appears to be conceptual or requires explanation"
            }
    
    def _execute_sql(self, query: str) -> Dict[str, Any]:
        """
        Execute query using SQL tool.
        
        Args:
            query: User query (will be converted to SQL)
            
        Returns:
            Dictionary with SQL results
        """
        print(f"\n{'='*60}")
        print("STEP 1: Generating SQL Query")
        print(f"{'='*60}")
        print("Converting user query to SQL...\n")
        
        # Get database schema
        schema = self.sql_tool.get_schema()
        
        # Generate SQL query using LLM
        try:
            sql_prompt = load_template(
                "agent.sql_generation",
                user_query=query,
                schema=schema
            )
        except FileNotFoundError:
            sql_prompt = f"""Convert the user query into a SQL SELECT query.

Database Schema:
{schema}

User Query: {query}

Generate a valid SQL SELECT query. Return ONLY the SQL query, nothing else.
"""
        
        sql_query = self.llm_client.generate(
            sql_prompt,
            system_prompt="You are a SQL query generator. Return only valid SQL SELECT statements.",
            temperature=0.1,
            max_tokens=300
        )
        
        # Clean SQL query (remove markdown code blocks if present)
        sql_query = sql_query.strip()
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sql_query
        sql_query = sql_query.rstrip("`")
        
        print(f"Generated SQL: {sql_query}\n")
        
        # Execute SQL query
        print(f"{'='*60}")
        print("STEP 2: Executing SQL Query")
        print(f"{'='*60}\n")
        
        sql_result = self.sql_tool.execute_query(sql_query)
        
        if not sql_result["success"]:
            print(f"SQL Error: {sql_result['error']}\n")
            return {
                "success": False,
                "answer": f"SQL query failed: {sql_result['error']}",
                "tool_used": "sql",
                "metadata": {
                    "sql_query": sql_query,
                    "error": sql_result["error"]
                }
            }
        
        results = sql_result["results"]
        
        if not results:
            answer = "No results found for the query."
        else:
            # Format results for display
            print(f"✓ Found {len(results)} result(s)\n")
            
            # Generate natural language answer from SQL results
            try:
                answer_prompt = load_template(
                    "agent.sql_answer",
                    user_query=query,
                    results=json.dumps(results, indent=2)
                )
            except FileNotFoundError:
                answer_prompt = f"""Based on the SQL query results, provide a natural language answer to the user's question.

User Query: {query}

SQL Results:
{json.dumps(results, indent=2)}

Provide a clear, concise answer based on these results.
"""
            
            answer = self.llm_client.generate(
                answer_prompt,
                system_prompt="You are a helpful assistant that explains SQL query results in natural language.",
                temperature=0.5,
                max_tokens=300
            )
        
        # Display results
        print("-"*60)
        print("SQL RESULTS:")
        print("-"*60)
        if results:
            # Print as table
            if results:
                headers = list(results[0].keys())
                print(f"{' | '.join(headers)}")
                print("-" * 60)
                for row in results[:10]:  # Show first 10 rows
                    values = [str(row.get(h, '')) for h in headers]
                    print(f"{' | '.join(values)}")
                if len(results) > 10:
                    print(f"... and {len(results) - 10} more rows")
        else:
            print("No results")
        print("-"*60)
        
        # Display answer
        print(f"\n{'='*60}")
        print("ANSWER")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "answer": answer,
            "tool_used": "sql",
            "metadata": {
                "sql_query": sql_query,
                "row_count": len(results)
            }
        }
    
    def _execute_rag(self, query: str) -> Dict[str, Any]:
        """
        Execute query using RAG tool.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with RAG results
        """
        print(f"\n{'='*60}")
        print("STEP 1: Retrieving Documents with RAG")
        print(f"{'='*60}\n")
        
        # Perform RAG search
        search_response = self.rag_client.search_keyword(query, top=3, use_pydantic=True)
        
        if not search_response or not search_response.results:
            return {
                "success": False,
                "answer": "No relevant documents found. Please ensure your search index is populated.",
                "tool_used": "rag",
                "metadata": {}
            }
        
        # Extract metadata
        metadata = extract_metadata_from_results(search_response.results, search_type="rag")
        metadata['num_documents'] = len(search_response.results)
        scores = [result.score for result in search_response.results]
        metadata['scores'] = scores
        
        # Get context documents
        context_docs = [result.document for result in search_response.results]
        
        # Generate answer
        print(f"{'='*60}")
        print("STEP 2: Generating Answer with Context")
        print(f"{'='*60}\n")
        
        answer = self.llm_client.generate_with_context(query, context_docs)
        
        # Display answer with metadata
        display_rag_answer(answer, metadata=metadata, retrieved_docs=search_response.results)
        
        return {
            "success": True,
            "answer": answer,
            "tool_used": "rag",
            "metadata": metadata
        }

