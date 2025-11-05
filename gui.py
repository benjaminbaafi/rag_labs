"""
Simple GUI for RAG Labs
======================
A user-friendly interface for running RAG labs without using the terminal.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from labs import (
    lab1_basic_rag,
    lab2_vector_rag,
    lab3_hybrid_rag,
    lab4_advanced_rag,
    lab5_setup_index,
    lab7_ai_agent,
    lab8_agent_assignment
)
import sys
import io


class RAGLabsGUI:
    """GUI application for RAG Labs."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Labs - Educational RAG Experiments")
        self.root.geometry("1000x700")
        
        # Variables
        self.selected_lab = tk.StringVar(value="1")
        self.query_text = tk.StringVar()
        self.custom_text = tk.StringVar()
        self.use_custom_text = tk.BooleanVar(value=False)
        self.run_comparison = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="RAG Labs - Retrieval-Augmented Generation Experiments",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Lab selection
        ttk.Label(main_frame, text="Select Lab:", font=("Arial", 10)).grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        
        lab_frame = ttk.Frame(main_frame)
        lab_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        labs = [
            ("1", "Lab 1: Basic RAG"),
            ("2", "Lab 2: Vector-Based RAG"),
            ("3", "Lab 3: Hybrid RAG"),
            ("4", "Lab 4: Advanced RAG"),
            ("5", "Lab 5: Setup Index"),
            ("6", "Lab 6: Semantic Search RAG"),
            ("7", "Lab 7: AI Agent (Complete)"),
            ("8", "Lab 8: AI Agent (Assignment)")
        ]
        
        for value, text in labs:
            ttk.Radiobutton(
                lab_frame,
                text=text,
                variable=self.selected_lab,
                value=value,
                command=self.on_lab_change
            ).pack(side=tk.LEFT, padx=5)
        
        # Query input
        ttk.Label(main_frame, text="Your Question:", font=("Arial", 10)).grid(
            row=2, column=0, sticky=(tk.W, tk.N), pady=5
        )
        
        query_entry = ttk.Entry(
            main_frame,
            textvariable=self.query_text,
            width=50,
            font=("Arial", 10)
        )
        query_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # Lab-specific options frame
        self.options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        self.options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Output area
        ttk.Label(main_frame, text="Results:", font=("Arial", 10, "bold")).grid(
            row=4, column=0, sticky=(tk.W, tk.N), pady=5
        )
        
        self.output_text = scrolledtext.ScrolledText(
            main_frame,
            width=80,
            height=20,
            font=("Consolas", 9),
            wrap=tk.WORD
        )
        self.output_text.grid(row=4, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.run_button = ttk.Button(
            button_frame,
            text="Run Lab",
            command=self.run_lab,
            width=20
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear Output",
            command=self.clear_output,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Exit",
            command=self.root.quit,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Initialize custom text widget (will be set in on_lab_change)
        self.custom_text_widget = None
        
        # Initialize options
        self.on_lab_change()
        
        # Set default query
        self.query_text.set("What is machine learning?")
        
    def on_lab_change(self):
        """Update options when lab selection changes."""
        # Clear options frame
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        
        lab_num = self.selected_lab.get()
        
        if lab_num == "1":
            # Lab 1: Custom text option
            ttk.Checkbutton(
                self.options_frame,
                text="Use custom text as context (no index search)",
                variable=self.use_custom_text
            ).pack(anchor=tk.W)
            
            ttk.Label(
                self.options_frame,
                text="Custom Text (will be used directly as context):",
                font=("Arial", 9)
            ).pack(anchor=tk.W, pady=(10, 5))
            
            custom_text_entry = scrolledtext.ScrolledText(
                self.options_frame,
                width=60,
                height=5,
                font=("Arial", 9),
                wrap=tk.WORD
            )
            custom_text_entry.pack(fill=tk.BOTH, expand=True, padx=5)
            custom_text_entry.insert("1.0", "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or classifications.")
            self.custom_text_widget = custom_text_entry
            
        elif lab_num == "2":
            # Lab 2: Comparison option
            ttk.Checkbutton(
                self.options_frame,
                text="Compare keyword vs. vector search",
                variable=self.run_comparison
            ).pack(anchor=tk.W)
            
        elif lab_num == "3":
            # Lab 3: Comparison option
            ttk.Checkbutton(
                self.options_frame,
                text="Compare all search methods (keyword, vector, hybrid)",
                variable=self.run_comparison
            ).pack(anchor=tk.W)
            
        elif lab_num == "4":
            # Lab 4: No additional options
            ttk.Label(
                self.options_frame,
                text="Advanced RAG with multi-step retrieval and context management",
                font=("Arial", 9),
                foreground="gray"
            ).pack(anchor=tk.W)
            
        elif lab_num == "5":
            # Lab 5: Setup index
            ttk.Label(
                self.options_frame,
                text="This will create/update the Azure AI Search index and upload sample documents.",
                font=("Arial", 9),
                foreground="gray"
            ).pack(anchor=tk.W)
    
    def clear_output(self):
        """Clear the output text area."""
        self.output_text.delete("1.0", tk.END)
    
    def write_output(self, text):
        """Write text to output area (thread-safe)."""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def run_lab(self):
        """Run the selected lab in a separate thread."""
        # Disable run button
        self.run_button.config(state="disabled")
        
        # Clear output
        self.clear_output()
        
        # Run in separate thread to avoid blocking UI
        thread = threading.Thread(target=self._run_lab_thread)
        thread.daemon = True
        thread.start()
    
    def _run_lab_thread(self):
        """Run lab in separate thread."""
        try:
            lab_num = self.selected_lab.get()
            query = self.query_text.get().strip()
            
            if not query and lab_num not in ["5"]:
                self.write_output("Error: Please enter a question.\n")
                self.root.after(0, lambda: self.run_button.config(state="normal"))
                return
            
            # Redirect stdout to capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                if lab_num == "1":
                    self._run_lab1(query)
                elif lab_num == "2":
                    self._run_lab2(query)
                elif lab_num == "3":
                    self._run_lab3(query)
                elif lab_num == "4":
                    self._run_lab4(query)
                elif lab_num == "5":
                    self._run_lab5()
                elif lab_num == "6":
                    self._run_lab6(query)
                elif lab_num == "7":
                    self._run_lab7(query)
                elif lab_num == "8":
                    self._run_lab8(query)
                
                # Get captured output
                output = captured_output.getvalue()
                
                # Write to GUI
                self.root.after(0, lambda: self.write_output(output))
                
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:  # pylint: disable=broad-except
            error_msg = f"Error: {str(e)}\n"
            self.root.after(0, lambda: self.write_output(error_msg))
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            # Re-enable run button
            self.root.after(0, lambda: self.run_button.config(state="normal"))
    
    def _run_lab1(self, query):
        """Run Lab 1."""
        custom_text = None
        if self.use_custom_text.get() and self.custom_text_widget:
            custom_text = self.custom_text_widget.get("1.0", tk.END).strip()
            if not custom_text:
                custom_text = """Machine learning is a subset of artificial intelligence that enables computers to learn 
and make decisions from data without being explicitly programmed. It uses algorithms to 
identify patterns in data and make predictions or classifications."""
        
        lab1_basic_rag.basic_rag(
            query,
            use_custom_text=self.use_custom_text.get(),
            custom_text=custom_text
        )
    
    def _run_lab2(self, query):
        """Run Lab 2."""
        lab2_vector_rag.vector_rag(query)
        
        if self.run_comparison.get():
            lab2_vector_rag.compare_keyword_vs_vector(query)
    
    def _run_lab3(self, query):
        """Run Lab 3."""
        lab3_hybrid_rag.hybrid_rag(query)
        
        if self.run_comparison.get():
            lab3_hybrid_rag.compare_all_methods(query)
    
    def _run_lab4(self, query):
        """Run Lab 4."""
        lab4_advanced_rag.advanced_rag(
            query,
            use_multi_step=True,
            use_context_management=True
        )
    
    def _run_lab5(self):
        """Run Lab 5."""
        lab5_setup_index.print_index_schema()
        lab5_setup_index.create_index()
        lab5_setup_index.upload_sample_documents()
    
    def _run_lab6(self, query):
        """Run Lab 6."""
        from labs import lab6_semantic_rag
        lab6_semantic_rag.semantic_rag(query)
        
        if self.run_comparison.get():
            lab6_semantic_rag.compare_all_search_methods(query)
    
    def _run_lab7(self, query):
        """Run Lab 7."""
        lab7_ai_agent.run_ai_agent(query)
    
    def _run_lab8(self, query):
        """Run Lab 8."""
        if query:
            lab8_agent_assignment.run_student_agent(query)
        else:
            lab8_agent_assignment.test_agent()


def main():
    """Main entry point for GUI."""
    root = tk.Tk()
    RAGLabsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

