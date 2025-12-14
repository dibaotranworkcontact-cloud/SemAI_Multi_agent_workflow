"""
RAG Tool for CrewAI Integration
Enables agents to search documentation and retrieve relevant context.
"""

from typing import Optional, List, Dict, Any
from crewai.tools import BaseTool
from semai.rag_system import RAGSystem, get_rag_system, initialize_rag


class RAGSearchTool(BaseTool):
    """Tool for semantic search over documentation and code"""
    
    name: str = "rag_search"
    description: str = (
        "Search documentation, code, and model information. "
        "Use this to find relevant information about models, APIs, configurations, "
        "and implementation details. "
        "Input: your search query (e.g., 'How do I use the Black-Scholes model?')"
    )
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        self.rag_system = rag_system
    
    def _run(self, query: str) -> str:
        """Execute search query"""
        if self.rag_system is None:
            return "RAG system not initialized. Please initialize RAG system first."
        
        try:
            results = self.rag_system.search(query, top_k=3)
            
            if not results:
                return f"No relevant documents found for: {query}"
            
            output = f"Found {len(results)} relevant document(s):\n\n"
            
            for i, result in enumerate(results, 1):
                output += (
                    f"{i}. {result['title']} (from {result['source']})\n"
                    f"   Relevance: {result['similarity']:.1%}\n"
                    f"   Preview: {result['content'][:200]}...\n\n"
                )
            
            return output
        except Exception as e:
            return f"Error during search: {str(e)}"


class RAGGetContextTool(BaseTool):
    """Tool to get full context for LLM augmentation"""
    
    name: str = "rag_get_context"
    description: str = (
        "Retrieve full context from relevant documents for detailed information. "
        "Use this when you need comprehensive information to complete a task. "
        "Input: your query or topic (e.g., 'Black-Scholes model implementation')"
    )
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        self.rag_system = rag_system
    
    def _run(self, query: str) -> str:
        """Get full context for query"""
        if self.rag_system is None:
            return "RAG system not initialized. Please initialize RAG system first."
        
        try:
            context = self.rag_system.get_context(query, top_k=3)
            return context
        except Exception as e:
            return f"Error retrieving context: {str(e)}"


class RAGModelLookupTool(BaseTool):
    """Tool to lookup specific models and their APIs"""
    
    name: str = "rag_model_lookup"
    description: str = (
        "Lookup specific derivative pricing models and their APIs. "
        "Input: model name (e.g., 'BlackScholesModel', 'LinearRegressionModel')"
    )
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        self.rag_system = rag_system
    
    def _run(self, model_name: str) -> str:
        """Lookup model information"""
        if self.rag_system is None:
            return "RAG system not initialized."
        
        try:
            # Search for the model in API documentation
            results = self.rag_system.search(model_name, top_k=5, doc_type="api")
            
            if not results:
                # Fallback to all documents
                results = self.rag_system.search(model_name, top_k=3)
            
            if not results:
                return f"Model '{model_name}' not found in documentation."
            
            output = f"Model Information for '{model_name}':\n\n"
            for result in results:
                output += f"{result['full_content']}\n\n"
            
            return output
        except Exception as e:
            return f"Error looking up model: {str(e)}"


class RAGGuardrailLookupTool(BaseTool):
    """Tool to lookup guardrail and safety information"""
    
    name: str = "rag_guardrail_lookup"
    description: str = (
        "Search for information about LLAMA guardrails and safety checks. "
        "Input: topic or feature (e.g., 'input validation', 'safety threshold')"
    )
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        self.rag_system = rag_system
    
    def _run(self, query: str) -> str:
        """Lookup guardrail information"""
        if self.rag_system is None:
            return "RAG system not initialized."
        
        try:
            # Search for guardrail-related content
            full_query = f"guardrail {query}"
            results = self.rag_system.search(full_query, top_k=3)
            
            if not results:
                results = self.rag_system.search(query, top_k=3)
            
            if not results:
                return f"No guardrail information found for: {query}"
            
            output = f"Guardrail Information for '{query}':\n\n"
            for result in results:
                output += f"{result['title']}:\n{result['full_content']}\n\n"
            
            return output
        except Exception as e:
            return f"Error looking up guardrail info: {str(e)}"


class RAGCodeExampleTool(BaseTool):
    """Tool to find code examples and implementations"""
    
    name: str = "rag_code_example"
    description: str = (
        "Find code examples and implementations. "
        "Input: what you want to implement (e.g., 'train a model', 'make predictions')"
    )
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        self.rag_system = rag_system
    
    def _run(self, query: str) -> str:
        """Find code examples"""
        if self.rag_system is None:
            return "RAG system not initialized."
        
        try:
            # Search code documents
            results = self.rag_system.search(query, top_k=3, doc_type="code")
            
            if not results:
                # Fallback to all documents
                results = self.rag_system.search(query, top_k=3)
            
            if not results:
                return f"No code examples found for: {query}"
            
            output = f"Code Examples for '{query}':\n\n"
            for result in results:
                output += f"File: {result['source']}\n{result['full_content']}\n\n"
            
            return output
        except Exception as e:
            return f"Error finding code examples: {str(e)}"


class RAGReferenceDocTool(BaseTool):
    """Tool to access reference documentation"""
    
    name: str = "rag_reference"
    description: str = (
        "Access reference documentation for concepts and algorithms. "
        "Input: concept or topic (e.g., 'neural networks', 'SDE', 'derivatives')"
    )
    
    def __init__(self, rag_system: Optional[RAGSystem] = None):
        super().__init__()
        self.rag_system = rag_system
    
    def _run(self, query: str) -> str:
        """Get reference documentation"""
        if self.rag_system is None:
            return "RAG system not initialized."
        
        try:
            # Prefer markdown documentation
            results = self.rag_system.search(query, top_k=3, doc_type="markdown")
            
            if not results:
                results = self.rag_system.search(query, top_k=3)
            
            if not results:
                return f"No reference documentation found for: {query}"
            
            output = f"Reference Documentation for '{query}':\n\n"
            for result in results:
                output += f"{result['title']}:\n{result['full_content']}\n\n"
            
            return output
        except Exception as e:
            return f"Error retrieving reference docs: {str(e)}"


def create_rag_tools(rag_system: Optional[RAGSystem] = None) -> List[BaseTool]:
    """
    Create all RAG tools for CrewAI agents.
    
    Args:
        rag_system: RAGSystem instance (if None, will use global instance)
        
    Returns:
        List of BaseTool instances
    """
    tools = [
        RAGSearchTool(rag_system),
        RAGGetContextTool(rag_system),
        RAGModelLookupTool(rag_system),
        RAGGuardrailLookupTool(rag_system),
        RAGCodeExampleTool(rag_system),
        RAGReferenceDocTool(rag_system)
    ]
    
    return tools


def create_rag_tools_for_agents(workspace_root: str) -> List[BaseTool]:
    """
    Create RAG tools with automatic initialization.
    
    Args:
        workspace_root: Path to workspace root
        
    Returns:
        List of BaseTool instances
    """
    # Initialize RAG system
    rag_system = initialize_rag(workspace_root)
    
    # Create and return tools
    return create_rag_tools(rag_system)
