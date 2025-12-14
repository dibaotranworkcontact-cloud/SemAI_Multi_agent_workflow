# RAG System Quick Start

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
pip install sentence-transformers scikit-learn
```

### Step 2: Initialize RAG System
```python
from semai.rag_system import RAGSystem

rag = RAGSystem(workspace_root=".")
rag.build_index()
print("✓ RAG system ready!")
```

### Step 3: Search for Information
```python
results = rag.search("Black-Scholes model", top_k=3)
for result in results:
    print(f"{result['title']}: {result['similarity']:.1%}")
```

### Step 4: Use in CrewAI Agents
```python
from semai.tools.rag_tools import create_rag_tools_for_agents
from crewai import Agent

tools = create_rag_tools_for_agents(".")
agent = Agent(
    role="Researcher",
    tools=tools
)
```

---

## 📚 Common Use Cases

### Use Case 1: Find Model Documentation
```python
rag = RAGSystem(".")
rag.build_index()

# Search for specific model
results = rag.search("LinearRegressionModel", doc_type="api")
print(results[0]['full_content'])
```

### Use Case 2: Get Examples
```python
# Find code examples
results = rag.search("train model", doc_type="code", top_k=3)
for r in results:
    print(f"Found in: {r['source']}")
    print(r['full_content'][:500])
```

### Use Case 3: Understand Concepts
```python
# Get full context for learning
context = rag.get_context("derivative pricing")
print(context)
```

### Use Case 4: Augment LLM Prompts
```python
user_question = "How do I implement a neural network?"

# Get relevant context
context = rag.get_context(user_question, top_k=3)

# Create augmented prompt
prompt = f"""
Based on this context:
{context}

Please answer: {user_question}
"""

# Send to LLM
response = llm.generate(prompt)
```

---

## 🔧 Configuration

### Custom Chunk Size
```python
from semai.rag_system import DocumentLoader

loader = DocumentLoader(chunk_size=256, overlap=25)
# Then use in RAGSystem
```

### Similarity Threshold
```python
from semai.rag_system import RAGRetriever

# Only return highly relevant results
retriever = RAGRetriever(documents, similarity_threshold=0.5)
```

### Different Embedding Model
```python
rag = RAGSystem(
    ".",
    embedding_model="all-mpnet-base-v2"  # More accurate, slower
)
```

---

## 🎯 Tips & Tricks

### 1. Better Search Results
```python
# Instead of: "model" (too generic)
results = rag.search("How to use BlackScholesModel for pricing")

# More specific queries give better results
```

### 2. Filter by Type
```python
# Search only in documentation
markdown_results = rag.search(query, doc_type="markdown")

# Search only in code
code_results = rag.search(query, doc_type="code")

# Search only in model APIs
api_results = rag.search(query, doc_type="api")
```

### 3. Check Relevance
```python
results = rag.search(query)
for r in results:
    if r['similarity'] > 0.5:  # High confidence
        print(f"High confidence: {r['title']}")
    elif r['similarity'] > 0.3:  # Medium confidence
        print(f"Medium confidence: {r['title']}")
```

### 4. Batch Operations
```python
# Search multiple queries
queries = ["model training", "validation", "evaluation"]
for q in queries:
    results = rag.search(q)
    print(f"Query: {q} → {len(results)} results")
```

---

## 📊 RAG Tools for Agents

| Tool | Use Case | Example |
|------|----------|---------|
| `rag_search` | Quick information lookup | "What is a guardrail?" |
| `rag_get_context` | In-depth research | "Explain neural networks" |
| `rag_model_lookup` | Find specific models | "Show BlackScholesModel" |
| `rag_guardrail_lookup` | Safety information | "What validation checks exist?" |
| `rag_code_example` | Implementation help | "How to train a model?" |
| `rag_reference` | Conceptual learning | "What is Black-Scholes?" |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No results found" | Use simpler query, lower similarity_threshold |
| "sentence-transformers not found" | `pip install sentence-transformers` |
| "Slow startup" | Use caching: `build_index(use_cache=True)` |
| "Memory issues" | Reduce chunk_size or use fewer documents |

---

## 📖 Full Guide

See [RAG_SYSTEM_GUIDE.md](RAG_SYSTEM_GUIDE.md) for:
- Complete API reference
- Advanced configuration
- Performance tuning
- Integration patterns
- Troubleshooting guide

---

## 🎓 Examples

Run working examples:
```bash
python rag_examples.py
```

Includes:
1. Basic initialization
2. Search by type
3. Context retrieval
4. Model lookup
5. Guardrail search
6. Code examples
7. Batch searching
8. Statistics
9. CrewAI integration
10. And more!

---

## 🚀 Integration with Crew

```python
from crewai import Agent, Task, Crew
from semai.tools.rag_tools import create_rag_tools_for_agents

# Initialize RAG tools
tools = create_rag_tools_for_agents(".")

# Create research agent
researcher = Agent(
    role="Research Specialist",
    goal="Find and explain derivative pricing concepts",
    tools=tools
)

# Create task
task = Task(
    description="Explain how to implement the Black-Scholes model",
    agent=researcher
)

# Run
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

---

## ✅ Verification

Check that RAG is working:
```python
from semai.rag_system import RAGSystem

rag = RAGSystem(".")
rag.build_index()

# Should return statistics
stats = rag.get_stats()
print(f"✓ RAG loaded {stats['total_documents']} documents")
print(f"✓ Embeddings enabled: {stats['has_embeddings']}")
```

---

## 📝 Next Steps

1. ✅ Install dependencies
2. ✅ Run `rag_examples.py`
3. ✅ Integrate RAG tools into your agents
4. ✅ Add RAG to your main workflow
5. ✅ Monitor with `get_stats()`
