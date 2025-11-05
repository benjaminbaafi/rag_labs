"""
RAG Labs - Main Entry Point
============================

Educational labs for learning RAG (Retrieval-Augmented Generation)
using Azure AI Search and OpenAI.

Run individual labs or use the menu to select a lab.
"""

import sys
from labs import (
    lab1_basic_rag,
    lab2_vector_rag,
    lab3_hybrid_rag,
    lab4_advanced_rag,
    lab5_setup_index,
    lab7_ai_agent,
    lab8_agent_assignment,
    lab9_weather_agent
)


def show_menu():
    """Display the lab menu."""
    print("\n" + "="*60)
    print("RAG LABS - Educational Labs for AI Engineers")
    print("="*60)
    print("\nAvailable Labs:")
    print("  1. Lab 1: Basic RAG (Retrieval-Augmented Generation)")
    print("  2. Lab 2: Vector-Based RAG (Vector Search)")
    print("  3. Lab 3: Hybrid RAG (Keyword + Vector)")
    print("  4. Lab 4: Advanced RAG (Multi-step, Re-ranking)")
    print("  5. Lab 5: Setup Azure AI Search Index")
    print("  6. Lab 6: Semantic Search RAG (AI-powered ranking)")
    print("  7. Lab 7: AI Agent (Complete Example - RAG + SQL)")
    print("  8. Lab 8: AI Agent (Take-Home Assignment)")
    print("  9. Lab 9: Weather Recommendation Agent")
    print("  0. Exit")
    print("\n" + "="*60)


def run_lab(lab_number: int):
    """Run the specified lab."""
    if lab_number == 1:
        print("\nRunning Lab 1: Basic RAG...")
        # Get user input
        query = input("\nEnter your question: ").strip()
        if not query:
            query = "What is machine learning?"  # Default if empty
            print(f"Using default query: {query}")
        
        # Ask about custom text
        use_custom = input("\nDo you want to use custom text? (y/n): ").strip().lower()
        print("(Custom text will be used directly as context - no index search)")
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
        
        if not custom_text:
            # Default custom text
            custom_text = """
            Machine learning is a subset of artificial intelligence that enables computers to learn 
            and make decisions from data without being explicitly programmed. It uses algorithms to 
            identify patterns in data and make predictions or classifications. Common types include 
            supervised learning (learning from labeled examples), unsupervised learning (finding 
            patterns in unlabeled data), and reinforcement learning (learning through trial and error).
            
            Deep learning is a specialized form of machine learning that uses neural networks with 
            multiple layers to process complex data. It's particularly effective for tasks like image 
            recognition, natural language processing, and speech recognition.
            """
        
        lab1_basic_rag.basic_rag(
            query, 
            use_custom_text=use_custom == 'y', 
            custom_text=custom_text
        )
    
    elif lab_number == 2:
        print("\nRunning Lab 2: Vector-Based RAG...")
        query = input("\nEnter your question: ").strip()
        if not query:
            query = "How does neural network training work?"
            print(f"Using default query: {query}")
        
        lab2_vector_rag.vector_rag(query)
        
        compare = input("\nCompare keyword vs. vector search? (y/n): ").strip().lower()
        if compare == 'y':
            lab2_vector_rag.compare_keyword_vs_vector(query)
    
    elif lab_number == 3:
        print("\nRunning Lab 3: Hybrid RAG...")
        query = input("\nEnter your question: ").strip()
        if not query:
            query = "Explain deep learning architectures"
            print(f"Using default query: {query}")
        
        lab3_hybrid_rag.hybrid_rag(query)
        
        compare = input("\nCompare all search methods? (y/n): ").strip().lower()
        if compare == 'y':
            lab3_hybrid_rag.compare_all_methods(query)
    
    elif lab_number == 4:
        print("\nRunning Lab 4: Advanced RAG...")
        query = input("\nEnter your question: ").strip()
        if not query:
            query = "What are the latest advances in transformer models?"
            print(f"Using default query: {query}")
        
        lab4_advanced_rag.advanced_rag(
            query,
            use_multi_step=True,
            use_context_management=True
        )
    
    elif lab_number == 5:
        print("\nRunning Lab 5: Setup Azure AI Search Index...")
        lab5_setup_index.print_index_schema()
        lab5_setup_index.create_index()
        lab5_setup_index.upload_sample_documents()
    
    elif lab_number == 6:
        print("\nRunning Lab 6: Semantic Search RAG...")
        query = input("\nEnter your question: ").strip()
        if not query:
            query = "What is machine learning?"
            print(f"Using default query: {query}")
        
        from labs import lab6_semantic_rag
        lab6_semantic_rag.semantic_rag(query)
        
        compare = input("\nCompare all search methods? (y/n): ").strip().lower()
        if compare == 'y':
            lab6_semantic_rag.compare_all_search_methods(query)
    
    elif lab_number == 7:
        print("\nRunning Lab 7: AI Agent (Complete Example)...")
        query = input("\nEnter your question: ").strip()
        if not query:
            query = "How many employees are in Engineering?"
            print(f"Using default query: {query}")
        
        lab7_ai_agent.run_ai_agent(query)
        
        demo = input("\nWould you like to see more examples? (y/n): ").strip().lower()
        if demo == 'y':
            lab7_ai_agent.demonstrate_agent()
    
    elif lab_number == 8:
        print("\nRunning Lab 8: AI Agent (Take-Home Assignment)...")
        print("\n⚠️  This is a take-home assignment!")
        print("Complete the TODO sections in StudentAgent class.\n")
        query = input("\nEnter your question (or press Enter to run tests): ").strip()
        
        if query:
            lab8_agent_assignment.run_student_agent(query)
        else:
            print("\nRunning test suite...")
            lab8_agent_assignment.test_agent()
    
    elif lab_number == 9:
        print("\nRunning Lab 9: Weather Recommendation Agent...")
        location = input("\nEnter location (or press Enter for 'New York'): ").strip()
        if not location:
            location = "New York"
            print(f"Using default location: {location}")
        
        # Get user email
        user_email = input("\nEnter your email address: ").strip()
        if not user_email:
            user_email = "user@example.com"
            print(f"Using default email: {user_email} (will only save to file)")
        
        # Ask if user wants to send email
        send_email = input("\nDo you want to send the email? (y/n): ").strip().lower()
        actually_send = send_email == 'y'
        
        if actually_send:
            print("\n📧 Email will be sent via SMTP")
            print("   Note: You'll need to provide sender email credentials next")
        else:
            print("\n💾 Email will only be saved to file")
        
        lab9_weather_agent.run_weather_agent(location, user_email, actually_send)
    
    else:
        print("Invalid lab number.")


def main():
    """Main entry point."""
    # Check for GUI flag
    if len(sys.argv) > 1 and sys.argv[1] in ["--gui", "-g", "gui"]:
        try:
            from gui import main as gui_main
            gui_main()
        except ImportError:
            print("Error: GUI requires tkinter. Please use Python with tkinter support.")
            print("You can still use the terminal interface with: python main.py")
        return
    
    if len(sys.argv) > 1:
        # Run specific lab from command line
        try:
            lab_number = int(sys.argv[1])
            if 1 <= lab_number <= 9:
                run_lab(lab_number)
            else:
                print("Invalid lab number. Please choose 1-9.")
        except ValueError:
            print("Please provide a valid lab number (1-9).")
    else:
        # Interactive menu
        while True:
            show_menu()
            try:
                choice = input("\nSelect a lab (0-9): ").strip()
                
                if choice == "0":
                    print("\nExiting. Happy learning!")
                    break
                
                lab_number = int(choice)
                if 1 <= lab_number <= 9:
                    run_lab(lab_number)
                else:
                    print("\nInvalid choice. Please select 0-9.")
            except ValueError:
                print("\nInvalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nExiting. Happy learning!")
                break
            except Exception as e:
                print(f"\nError running lab: {e}")
                print("Make sure you have:")
                print("  1. Set up your .env file with Azure credentials")
                print("  2. Created your Azure AI Search index (run Lab 5 first)")
                print("  3. Installed all dependencies")


if __name__ == "__main__":
    main()
