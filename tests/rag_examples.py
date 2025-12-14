"""
RAG System Examples
Demonstrates all major features of the RAG system.
"""

from pathlib import Path
import sys

# Add src to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "src"))

from semai.rag_system import RAGSystem, initialize_rag, RAGEmbedder, DocumentLoader
from semai.tools.rag_tools import create_rag_tools_for_agents


def example_1_basic_initialization():
    """Example 1: Basic RAG initialization and search"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic RAG Initialization and Search")
    print("="*70)
    
    try:
        # Initialize RAG system
        rag = RAGSystem(workspace_root=".")
        
        print("Building index...")
        doc_count = rag.build_index(use_cache=True)
        print(f"✓ Loaded {doc_count} documents")
        
        # Search for something
        query = "Black-Scholes model"
        print(f"\nSearching for: '{query}'")
        
        results = rag.search(query, top_k=3)
        
        print(f"✓ Found {len(results)} results\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Source: {result['source']}")
            print(f"   Relevance: {result['similarity']:.1%}")
            print(f"   Preview: {result['content'][:150]}...")
            print()
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_2_search_by_type():
    """Example 2: Search filtered by document type"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Search by Document Type")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        # Search in code documentation
        print("Searching in CODE documentation for 'neural network':")
        code_results = rag.search("neural network", top_k=2, doc_type="code")
        print(f"✓ Found {len(code_results)} code examples\n")
        for result in code_results:
            print(f"  • {result['title']} ({result['source']})")
        
        # Search in markdown
        print("\nSearching in MARKDOWN documentation for 'neural network':")
        md_results = rag.search("neural network", top_k=2, doc_type="markdown")
        print(f"✓ Found {len(md_results)} markdown docs\n")
        for result in md_results:
            print(f"  • {result['title']} ({result['source']})")
        
        # Search in API docs
        print("\nSearching in API documentation for 'model':")
        api_results = rag.search("model", top_k=2, doc_type="api")
        print(f"✓ Found {len(api_results)} API docs\n")
        for result in api_results:
            print(f"  • {result['title']} ({result['source']})")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_3_get_context():
    """Example 3: Get full context for LLM augmentation"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Get Full Context for LLM Augmentation")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        query = "gradient descent optimization"
        print(f"Getting context for: '{query}'\n")
        
        context = rag.get_context(query, top_k=2)
        
        print("Context retrieved:")
        print("-" * 70)
        print(context[:500] + "...")
        print("-" * 70)
        print(f"\n✓ Context retrieved ({len(context)} characters)")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_4_model_lookup():
    """Example 4: Lookup specific models"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Model Lookup")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        model_names = ["BlackScholesModel", "LinearRegressionModel", "DeepLearningNet"]
        
        for model_name in model_names:
            print(f"\nLooking up: {model_name}")
            results = rag.search(model_name, top_k=1, doc_type="api")
            
            if results:
                result = results[0]
                print(f"✓ Found: {result['title']}")
                print(f"  Source: {result['source']}")
                print(f"  Relevance: {result['similarity']:.1%}")
            else:
                print(f"✗ Not found in API docs")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_5_guardrail_search():
    """Example 5: Search guardrail and safety information"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Guardrail and Safety Search")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        queries = [
            "input validation",
            "safety checks",
            "prediction validation"
        ]
        
        for query in queries:
            print(f"\nSearching for: '{query}'")
            results = rag.search(query, top_k=2)
            
            print(f"✓ Found {len(results)} results")
            for result in results:
                print(f"  • {result['title']} - {result['similarity']:.1%} match")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_6_code_examples():
    """Example 6: Find code examples and implementations"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Code Examples and Implementations")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        tasks = [
            "train a model",
            "make predictions",
            "validate data"
        ]
        
        for task in tasks:
            print(f"\nFinding examples for: '{task}'")
            results = rag.search(task, top_k=2, doc_type="code")
            
            print(f"✓ Found {len(results)} code examples")
            for result in results:
                print(f"  • {result['source']}: {result['title']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_7_retrieve_by_source():
    """Example 7: Retrieve all documents from a specific source"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Retrieve by Source")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        # Get all documents from builtin_models.py
        source = "builtin_models.py"
        print(f"Retrieving all documents from: {source}")
        
        docs = rag.retriever.retrieve_by_source(source)
        print(f"✓ Found {len(docs)} documents\n")
        
        for doc in docs[:5]:  # Show first 5
            print(f"  • {doc.title} (chunk {doc.chunk_index + 1}/{doc.total_chunks})")
        
        if len(docs) > 5:
            print(f"  ... and {len(docs) - 5} more")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_8_batch_search():
    """Example 8: Batch searching multiple queries"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Batch Searching")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        queries = [
            "Black-Scholes pricing",
            "neural network training",
            "guardrail validation",
            "data preprocessing",
            "model evaluation"
        ]
        
        print(f"Searching for {len(queries)} queries...\n")
        
        all_results = {}
        for query in queries:
            results = rag.search(query, top_k=1)
            if results:
                all_results[query] = results[0]
        
        print(f"✓ Found results for {len(all_results)} queries\n")
        
        for query, result in all_results.items():
            print(f"Query: '{query}'")
            print(f"  → {result['title']} ({result['similarity']:.1%} match)")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_9_statistics():
    """Example 9: Get RAG system statistics"""
    print("\n" + "="*70)
    print("EXAMPLE 9: RAG System Statistics")
    print("="*70)
    
    try:
        rag = RAGSystem(workspace_root=".")
        rag.build_index(use_cache=True)
        
        stats = rag.get_stats()
        
        print(f"Total documents: {stats['total_documents']}")
        print(f"\nDocuments by type:")
        for doc_type, count in stats['by_type'].items():
            print(f"  • {doc_type}: {count}")
        
        print(f"\nEmbeddings:")
        print(f"  • Enabled: {stats['has_embeddings']}")
        print(f"  • Model: {stats['embedding_model']}")
        
        print(f"\nCache:")
        print(f"  • Location: {stats['cache_path']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_10_rag_tools():
    """Example 10: Using RAG tools with CrewAI"""
    print("\n" + "="*70)
    print("EXAMPLE 10: RAG Tools for CrewAI Agents")
    print("="*70)
    
    try:
        print("Creating RAG tools for agents...")
        tools = create_rag_tools_for_agents(workspace_root=".")
        
        print(f"✓ Created {len(tools)} RAG tools:\n")
        
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
            print(f"   Description: {tool.description[:80]}...")
            print()
        
        print("These tools can be used with CrewAI agents:")
        print("  agent = Agent(tools=tools)")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def run_all_examples():
    """Run all examples"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "RAG SYSTEM EXAMPLES" + " "*34 + "║")
    print("╚" + "="*68 + "╝")
    
    examples = [
        ("Basic Initialization", example_1_basic_initialization),
        ("Search by Type", example_2_search_by_type),
        ("Get Context", example_3_get_context),
        ("Model Lookup", example_4_model_lookup),
        ("Guardrail Search", example_5_guardrail_search),
        ("Code Examples", example_6_code_examples),
        ("Retrieve by Source", example_7_retrieve_by_source),
        ("Batch Search", example_8_batch_search),
        ("Statistics", example_9_statistics),
        ("CrewAI Tools", example_10_rag_tools),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example {i} ({name}) failed: {e}")
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run all examples
    run_all_examples()
    
    # Or run individual examples
    # example_1_basic_initialization()
    # example_2_search_by_type()
    # etc.
