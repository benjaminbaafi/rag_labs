"""
SQL Tool for AI Agent
====================
Provides a SQL query interface for the AI agent to query structured data.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class SQLTool:
    """SQL tool for querying structured data."""
    
    def __init__(self, db_path: str = "data/agent_database.db"):
        """
        Initialize SQL tool.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure database exists with sample schema."""
        if not self.db_path.exists():
            self._create_sample_database()
    
    def _create_sample_database(self):
        """Create a sample database with employee and product data."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create employees table
        cursor.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL,
                hire_date TEXT,
                email TEXT
            )
        """)
        
        # Create products table
        cursor.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL,
                stock INTEGER,
                description TEXT
            )
        """)
        
        # Create sales table
        cursor.execute("""
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                employee_id INTEGER,
                sale_date TEXT,
                quantity INTEGER,
                total_amount REAL,
                FOREIGN KEY (product_id) REFERENCES products(id),
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        """)
        
        # Insert sample employees
        employees = [
            ("Alice Johnson", "Engineering", 120000, "2020-01-15", "alice@company.com"),
            ("Bob Smith", "Sales", 95000, "2019-03-20", "bob@company.com"),
            ("Carol White", "Engineering", 115000, "2021-06-10", "carol@company.com"),
            ("David Brown", "Marketing", 88000, "2020-11-05", "david@company.com"),
            ("Eva Davis", "Sales", 92000, "2022-02-14", "eva@company.com"),
            ("Frank Miller", "Engineering", 125000, "2018-09-01", "frank@company.com"),
        ]
        cursor.executemany(
            "INSERT INTO employees (name, department, salary, hire_date, email) VALUES (?, ?, ?, ?, ?)",
            employees
        )
        
        # Insert sample products
        products = [
            ("Laptop Pro", "Electronics", 1299.99, 45, "High-performance laptop for professionals"),
            ("Wireless Mouse", "Electronics", 29.99, 150, "Ergonomic wireless mouse"),
            ("Office Chair", "Furniture", 299.99, 30, "Comfortable ergonomic office chair"),
            ("Desk Lamp", "Furniture", 49.99, 80, "LED desk lamp with adjustable brightness"),
            ("Monitor 27in", "Electronics", 399.99, 25, "27-inch 4K monitor"),
        ]
        cursor.executemany(
            "INSERT INTO products (name, category, price, stock, description) VALUES (?, ?, ?, ?, ?)",
            products
        )
        
        # Insert sample sales
        sales = [
            (1, 2, "2024-01-15", 2, 2599.98),
            (2, 2, "2024-01-16", 5, 149.95),
            (3, 4, "2024-01-17", 1, 299.99),
            (4, 5, "2024-01-18", 3, 149.97),
            (1, 5, "2024-01-19", 1, 1299.99),
            (5, 2, "2024-01-20", 2, 799.98),
        ]
        cursor.executemany(
            "INSERT INTO sales (product_id, employee_id, sale_date, quantity, total_amount) VALUES (?, ?, ?, ?, ?)",
            sales
        )
        
        conn.commit()
        conn.close()
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary with query results and metadata
        """
        try:
            # Security: Only allow SELECT queries
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT"):
                return {
                    "success": False,
                    "error": "Only SELECT queries are allowed for security reasons",
                    "results": None
                }
            
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Return rows as dicts
            cursor = conn.cursor()
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert rows to list of dicts
            results = [dict(row) for row in rows]
            
            conn.close()
            
            return {
                "success": True,
                "error": None,
                "results": results,
                "row_count": len(results)
            }
            
        except sqlite3.Error as e:
            return {
                "success": False,
                "error": str(e),
                "results": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "results": None
            }
    
    def get_schema(self) -> str:
        """
        Get database schema information.
        
        Returns:
            String describing the database schema
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        schema_info = []
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            schema_info.append(f"\nTable: {table_name}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                schema_info.append(f"  - {col_name} ({col_type})")
        
        conn.close()
        
        return "\n".join(schema_info)
    
    def describe_tool(self) -> str:
        """
        Get a description of this tool for the agent.
        
        Returns:
            Tool description string
        """
        return """
SQL Tool - Query structured data from a SQLite database.

Use this tool when:
- User asks for specific facts, numbers, or statistics
- User wants to query structured data (employees, products, sales, etc.)
- User asks "how many", "what is the", "list all", "find", etc.
- User needs exact matches or calculations

Database contains:
- employees: Employee information (name, department, salary, hire_date, email)
- products: Product catalog (name, category, price, stock, description)
- sales: Sales transactions (product_id, employee_id, sale_date, quantity, total_amount)

Examples:
- "How many employees are in Engineering?"
- "What is the total sales amount for January?"
- "List all products in the Electronics category"
- "Find employees with salary above 100000"
"""

