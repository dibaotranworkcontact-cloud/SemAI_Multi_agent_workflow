# RAG (Retrieval-Augmented Generation) System Guide

## Overview

The RAG system provides semantic search and retrieval capabilities for your derivative pricing system. It enables agents and applications to find relevant documentation, code examples, and model information quickly and accurately.

### Key Features

✅ **Semantic Search** - Find relevant content using natural language queries
✅ **Multi-Source Indexing** - Index documentation, code, and model metadata
✅ **Flexible Retrieval** - Search by content, type, or source
✅ **CrewAI Integration** - 6 specialized tools for agents
✅ **Automatic Caching** - Cache embeddings for fast initialization
✅ **Fallback Search** - Works with or without embeddings
✅ **Context Augmentation** - Get full context for LLM operations

---

## Architecture

### Components

#### 1. DocumentLoader
Loads and chunks documents from various sources:
- Markdown files (documentation)
- Python files (code and docstrings)
- Model metadata (from builtin_models.py)
- Custom documents

**Chunking Strategy:**
- Default chunk size: 512 characters
- Overlap: 50 characters (for context continuity)
- Automatic splitting for large documents

#### 2. RAGEmbedder
Creates semantic embeddings using sentence-transformers:
- Default model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Fast and lightweight
- Can be replaced with larger models for production

#### 3. RAGRetriever
Performs semantic and keyword search:
- Cosine similarity for semantic search
- Keyword overlap for fallback search
- Filtering by document type
- Configurable similarity threshold (default: 0.3)

#### 4. RAGSystem
Main orchestration class:
- Manages the full pipeline
- Handles document loading
- Manages caching
- Provides search interface

---

## Installation

### 1. Install Required Packages

```bash
pip install sentence-transformers scikit-learn
```

Or install from the project:

```bash
uv pip install sentence-transformers scikit-learn
```

### 2. Verify Installation

```python
from semai.rag_system import RAGSystem

# Will show warnings if dependencies are missing
system = RAGSystem(".")
```

---

## Usage

### Basic Usage

#### 1. Initialize RAG System

```python
from semai.rag_system import RAGSystem

# Create RAG system
rag = RAGSystem(workspace_root="/path/to/workspace")

# Build index (loads documents and embeddings)
doc_count = rag.build_index(
    load_markdown=True,
    load_code=True,
    load_models=True,
    use_cache=True
)

print(f"Loaded {doc_count} documents")
```

#### 2. Search for Information

```python
# Simple search
results = rag.search("How to use Black-Scholes model?", top_k=5)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Source: {result['source']}")
    print(f"Relevance: {result['similarity']:.1%}")
    print(f"Content: {result['content']}")
    print("-" * 50)
```

#### 3. Filter by Document Type

```python
# Search only in code documentation
code_results = rag.search(
    "neural network implementation",
    top_k=3,
    doc_type="code"
)

# Search only in API documentation
api_results = rag.search(
    "model training parameters",
    top_k=3,
    doc_type="api"
)

# Search in reference documentation
ref_results = rag.search(
    "Black-Scholes formula",
    top_k=3,
    doc_type="markdown"
)
```

#### 4. Get Full Context

```python
# Get formatted context for LLM augmentation
context = rag.get_context("gradient descent optimization", top_k=3)

# Use in LLM prompt
prompt = f"""
Based on the following context, explain the gradient descent algorithm:

{context}
"""
```

#### 5. Retrieve from Specific Source

```python
# Get all documents from a specific file
docs = rag.retriever.retrieve_by_source("builtin_models.py")

# Get all documents of a specific type
code_docs = rag.retriever.retrieve_by_type("code")
```

---

## CrewAI Integration

### Using RAG Tools in Agents

#### 1. Initialize RAG Tools

```python
from semai.tools.rag_tools import create_rag_tools_for_agents

# Create RAG tools (auto-initializes RAG system)
rag_tools = create_rag_tools_for_agents(workspace_root=".")

# Use in your agents
agent = Agent(
    role="Research Assistant",
    goal="Find information about derivative pricing models",
    tools=rag_tools
)
```

#### 2. Available RAG Tools

| Tool | Purpose | Example Query |
|------|---------|----------------|
| `rag_search` | General semantic search | "How to validate input data?" |
| `rag_get_context` | Get full context | "Black-Scholes pricing model" |
| `rag_model_lookup` | Find specific models | "BlackScholesModel" |
| `rag_guardrail_lookup` | Safety information | "input validation checks" |
| `rag_code_example` | Code examples | "train a model with guardrails" |
| `rag_reference` | Reference docs | "neural networks" |

#### 3. Example Agent Using RAG

```python
from crewai import Agent, Task, Crew
from semai.tools.rag_tools import create_rag_tools_for_agents

# Initialize RAG tools
rag_tools = create_rag_tools_for_agents(".")

# Create research agent
researcher = Agent(
    role="Finance Researcher",
    goal="Find and explain derivative pricing concepts",
    backstory="You are an expert in financial derivatives.",
    tools=rag_tools
)

# Create task
task = Task(
    description="Explain how the Black-Scholes model is implemented in our system",
    agent=researcher
)

# Create and run crew
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
```

---

## Document Types

### 1. Markdown (`markdown`)
Documentation files explaining concepts and features.

**Examples:**
- `LLAMA_GUARDRAIL_INTEGRATION.md`
- `README.md`
- `neural_network_lv_reference.md`

**Best for:** Understanding concepts, learning about features

### 2. Code (`code`)
Python source code with docstrings and comments.

**Sources:**
- `builtin_models.py`
- `llm_config.py`
- `crew.py`
- Model implementations

**Best for:** Implementation details, code patterns

### 3. API (`api`)
Model and class API documentation extracted from docstrings.

**Examples:**
- BlackScholesModel API
- LinearRegressionModel API
- RAGSystem API

**Best for:** Understanding APIs, method signatures

### 4. Reference (`reference`)
Custom reference documents.

**Best for:** Specialized information, domain knowledge

---

## Performance Tuning

### 1. Chunk Size

```python
# Smaller chunks (more granular retrieval)
loader = DocumentLoader(chunk_size=256, overlap=25)

# Larger chunks (more context per result)
loader = DocumentLoader(chunk_size=1024, overlap=100)
```

### 2. Similarity Threshold

```python
# Lower threshold (more results, lower precision)
retriever = RAGRetriever(documents, embedder, similarity_threshold=0.2)

# Higher threshold (fewer results, higher precision)
retriever = RAGRetriever(documents, embedder, similarity_threshold=0.5)
```

### 3. Embedding Model

```python
# Faster model (less accurate)
rag = RAGSystem(
    workspace_root=".",
    embedding_model="all-MiniLM-L6-v2"  # 384-dim, fast
)

# More accurate model (slower)
rag = RAGSystem(
    workspace_root=".",
    embedding_model="all-mpnet-base-v2"  # 768-dim, more accurate
)

# Production model (very accurate)
rag = RAGSystem(
    workspace_root=".",
    embedding_model="all-roberta-large-v1"  # 1024-dim, very accurate
)
```

### 4. Caching

```python
# Use cache (faster initialization on subsequent runs)
rag.build_index(use_cache=True)

# Skip cache (rebuild everything)
rag.build_index(use_cache=False)

# Clear cache
import shutil
shutil.rmtree(rag.cache_path)
```

---

## Advanced Features

### 1. Custom Documents

Add your own documents to the index:

```python
custom_docs = [
    {
        "content": "Custom pricing model documentation...",
        "title": "Custom Model",
        "source": "custom_docs.txt",
        "type": "reference",
        "metadata": {"custom": True}
    }
]

docs = loader.load_custom_documents(custom_docs)
```

### 2. Batch Searching

```python
queries = [
    "Black-Scholes model",
    "neural network training",
    "safety guardrails"
]

results = {}
for query in queries:
    results[query] = rag.search(query, top_k=3)
```

### 3. Filtering Results

```python
# Search with multiple filters
results = rag.search("training", top_k=10, doc_type="code")

# Post-filter results
filtered = [r for r in results if "model" in r['title']]
```

### 4. Getting Statistics

```python
stats = rag.get_stats()

print(f"Total documents: {stats['total_documents']}")
print(f"By type: {stats['by_type']}")
print(f"Has embeddings: {stats['has_embeddings']}")
print(f"Embedding model: {stats['embedding_model']}")
```

---

## Troubleshooting

### Issue: "sentence-transformers not installed"

**Solution:**
```bash
pip install sentence-transformers
```

### Issue: "No relevant documents found"

**Solutions:**
1. Check if index was built: `rag.build_index()`
2. Try different query terms
3. Lower similarity threshold: `RAGRetriever(..., similarity_threshold=0.2)`
4. Check document type filter: `doc_type=None` for all types

### Issue: Slow initialization

**Solutions:**
1. Use caching: `build_index(use_cache=True)`
2. Use faster embedding model: `all-MiniLM-L6-v2` (default)
3. Load fewer documents: `load_code=False` or `load_markdown=False`

### Issue: Out of memory

**Solutions:**
1. Reduce chunk size: `DocumentLoader(chunk_size=256)`
2. Use CPU-only embeddings (automatic)
3. Load documents in batches

---

## Best Practices

### 1. Query Formulation

✅ **Good queries:**
- "How do I train a Black-Scholes model?"
- "What are the input validation checks?"
- "Show me the neural network implementation"

❌ **Poor queries:**
- "Model"
- "Training"
- "Code"

### 2. Result Interpretation

```python
results = rag.search(query)

# Always check relevance score
for result in results:
    if result['similarity'] > 0.5:  # High confidence
        # Use this result
        pass
```

### 3. Cache Management

```python
# First run: Build and cache
rag = RAGSystem(".")
rag.build_index(use_cache=True)

# Subsequent runs: Load from cache (fast)
rag = RAGSystem(".")
rag.build_index(use_cache=True)

# When documents change: Skip cache
rag.build_index(use_cache=False)
```

### 4. Integration with LLM

```python
# Get context for LLM augmentation
context = rag.get_context(user_query, top_k=3)

# Create augmented prompt
augmented_prompt = f"""
You are an expert in derivative pricing.

Here is relevant context:
{context}

Now answer this question: {user_query}
"""

# Send to LLM
response = llm.generate(augmented_prompt)
```

---

## API Reference

### RAGSystem

```python
class RAGSystem:
    def __init__(
        self,
        workspace_root: str,
        use_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2"
    )
    
    def build_index(
        self,
        load_markdown: bool = True,
        load_code: bool = True,
        load_models: bool = True,
        use_cache: bool = True
    ) -> int
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]
    
    def get_context(
        self,
        query: str,
        top_k: int = 3
    ) -> str
    
    def get_stats(self) -> Dict[str, Any]
```

### RAGRetriever

```python
class RAGRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None
    ) -> List[Tuple[Document, float]]
    
    def retrieve_by_source(self, source: str) -> List[Document]
    
    def retrieve_by_type(self, doc_type: str) -> List[Document]
```

### RAG Tools

```python
def create_rag_tools(rag_system: Optional[RAGSystem] = None) -> List[BaseTool]
def create_rag_tools_for_agents(workspace_root: str) -> List[BaseTool]
```

---

## Examples

See [rag_examples.py](rag_examples.py) for 10 complete working examples:

1. Basic initialization and search
2. Searching by document type
3. Getting full context
4. Retrieving model information
5. Accessing guardrail documentation
6. Finding code examples
7. Using RAG with CrewAI agents
8. Batch searching
9. Custom document loading
10. Statistics and monitoring

---

## Performance Metrics

### Typical Performance

| Operation | Time | Notes |
|-----------|------|-------|
| First initialization | 30-60s | Building embeddings |
| Subsequent initialization | 1-2s | Loading from cache |
| Single search | 50-100ms | Semantic search |
| Batch search (10 queries) | 500-800ms | ~50-80ms per query |
| Context retrieval | 100-150ms | Full document retrieval |

### Index Size

- Total documents: ~200-300 (all markdown + code)
- Embedding size: ~2-5 MB (384-dim)
- Total cache: ~3-8 MB (documents + embeddings)

---

## Future Enhancements

- [ ] FAISS vector store for larger indexes
- [ ] Multi-language support
- [ ] Hybrid search (semantic + keyword)
- [ ] Real-time document indexing
- [ ] Query expansion and refinement
- [ ] Document clustering
- [ ] Citation tracking
- [ ] Custom similarity metrics

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review examples in [rag_examples.py](rag_examples.py)
3. Check RAG system statistics: `rag.get_stats()`
4. Enable verbose logging: `RAGSystem(..., verbose=True)`
