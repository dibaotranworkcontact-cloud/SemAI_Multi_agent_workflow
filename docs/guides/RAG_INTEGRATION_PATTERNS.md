# RAG Integration Patterns & Best Practices

## Overview

This document covers advanced integration patterns for the RAG system with your derivative pricing workflow.

---

## Pattern 1: Agent-Based RAG

### Use Case
Agents need to research and understand domain concepts to complete tasks.

### Implementation

```python
from crewai import Agent, Task, Crew
from semai.tools.rag_tools import create_rag_tools_for_agents

# Initialize RAG tools
rag_tools = create_rag_tools_for_agents(".")

# Create research agent
researcher = Agent(
    role="Derivative Pricing Researcher",
    goal="Find and explain derivative pricing concepts",
    backstory="Expert in financial derivatives and pricing models",
    tools=rag_tools,
    verbose=True
)

# Create task
task = Task(
    description="Explain the Black-Scholes model implementation",
    agent=researcher,
    expected_output="Detailed explanation of Black-Scholes implementation"
)

# Execute
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

### When to Use
- Research tasks that need background information
- Learning and documentation tasks
- Debugging and troubleshooting
- Concept explanation tasks

---

## Pattern 2: Context Augmentation

### Use Case
Enhance LLM prompts with relevant context from your knowledge base.

### Implementation

```python
from semai.rag_system import RAGSystem
from semai.llm_integration import get_llm

# Initialize RAG
rag = RAGSystem(".")
rag.build_index()

# Get user question
user_question = "How do I train a neural network for pricing?"

# Augment with context
context = rag.get_context(user_question, top_k=3)

# Create augmented prompt
prompt = f"""You are an expert in derivative pricing.

Here is relevant context from the knowledge base:
{context}

Now, please answer this question with detailed implementation steps:
{user_question}

Provide clear, actionable steps with code examples where applicable.
"""

# Get response from LLM
llm = get_llm()
response = llm.generate(prompt, max_tokens=2000)

print(response)
```

### When to Use
- Generate comprehensive documentation
- Create implementation guides
- Explain complex concepts
- Provide detailed code examples

---

## Pattern 3: Multi-Stage Retrieval

### Use Case
Retrieve increasingly specific information in stages.

### Implementation

```python
from semai.rag_system import RAGSystem

rag = RAGSystem(".")
rag.build_index()

# Stage 1: Broad search
general_results = rag.search("neural networks", top_k=5)
print("General concept results:", len(general_results))

# Stage 2: Specific search
specific_results = rag.search("neural network training implementation", top_k=3)
print("Specific implementation results:", len(specific_results))

# Stage 3: Code examples
code_examples = rag.search("train neural network", doc_type="code", top_k=2)
print("Code examples:", len(code_examples))

# Combine results
all_info = general_results + specific_results + code_examples
```

### When to Use
- Complex research questions
- Progressive learning paths
- Comprehensive understanding needed
- Building training materials

---

## Pattern 4: Model-Specific Lookup

### Use Case
Query all information about a specific model.

### Implementation

```python
from semai.rag_system import RAGSystem

rag = RAGSystem(".")
rag.build_index()

def get_model_info(model_name: str) -> dict:
    """Get comprehensive info about a model"""
    
    info = {
        "api_docs": [],
        "implementation": [],
        "examples": [],
        "guardrails": []
    }
    
    # API documentation
    api_docs = rag.search(model_name, doc_type="api", top_k=3)
    info["api_docs"] = api_docs
    
    # Implementation details
    impl = rag.search(f"{model_name} implementation", doc_type="code", top_k=2)
    info["implementation"] = impl
    
    # Code examples
    examples = rag.search(f"use {model_name}", doc_type="code", top_k=2)
    info["examples"] = examples
    
    # Guardrail information
    guardrails = rag.search(f"{model_name} validation", top_k=2)
    info["guardrails"] = guardrails
    
    return info

# Use it
model_info = get_model_info("BlackScholesModel")
print(f"Found {len(model_info['api_docs'])} API docs")
print(f"Found {len(model_info['examples'])} examples")
```

### When to Use
- Need comprehensive model information
- Building model selection tools
- Creating model documentation
- Understanding model capabilities

---

## Pattern 5: Concept-Based Learning Path

### Use Case
Create a structured learning path through related concepts.

### Implementation

```python
from semai.rag_system import RAGSystem

rag = RAGSystem(".")
rag.build_index()

def create_learning_path(topic: str, depth: int = 3) -> list:
    """Create a structured learning path"""
    
    path = []
    
    # 1. Conceptual understanding
    concepts = rag.search(topic, doc_type="markdown", top_k=depth)
    path.append(("Concepts", concepts))
    
    # 2. Theoretical foundation
    theory = rag.search(f"{topic} theory foundations", top_k=depth)
    path.append(("Theory", theory))
    
    # 3. Implementation details
    impl = rag.search(f"implement {topic}", doc_type="code", top_k=depth)
    path.append(("Implementation", impl))
    
    # 4. Practical examples
    examples = rag.search(f"{topic} example code", doc_type="code", top_k=depth)
    path.append(("Examples", examples))
    
    # 5. Best practices
    practices = rag.search(f"{topic} best practices", top_k=depth)
    path.append(("Best Practices", practices))
    
    return path

# Use it
learning_path = create_learning_path("Black-Scholes pricing")
for stage, materials in learning_path:
    print(f"\n{stage}:")
    for material in materials:
        print(f"  - {material['title']}")
```

### When to Use
- Educational content generation
- Training program creation
- Onboarding new team members
- Self-guided learning paths

---

## Pattern 6: Comparison and Analysis

### Use Case
Compare different models, approaches, or implementations.

### Implementation

```python
from semai.rag_system import RAGSystem

rag = RAGSystem(".")
rag.build_index()

def compare_models(models: list) -> dict:
    """Compare multiple models"""
    
    comparison = {}
    
    for model in models:
        comparison[model] = {
            "description": rag.search(model, top_k=1),
            "pros": rag.search(f"{model} advantages", top_k=2),
            "cons": rag.search(f"{model} limitations", top_k=2),
            "use_cases": rag.search(f"{model} use cases", top_k=2)
        }
    
    return comparison

# Compare models
models_to_compare = [
    "BlackScholesModel",
    "NeuralNetworkSDE",
    "DeepLearningNet"
]

comparison = compare_models(models_to_compare)

print("Model Comparison:")
for model, info in comparison.items():
    print(f"\n{model}:")
    if info["description"]:
        print(f"  Description: {info['description'][0]['title']}")
    print(f"  Advantages: {len(info['pros'])} docs")
    print(f"  Limitations: {len(info['cons'])} docs")
```

### When to Use
- Model selection decisions
- Feature comparison
- Architecture analysis
- Performance evaluation

---

## Pattern 7: Validation and Safety

### Use Case
Query safety and validation information for compliance.

### Implementation

```python
from semai.rag_system import RAGSystem

rag = RAGSystem(".")
rag.build_index()

def get_safety_requirements(component: str) -> dict:
    """Get all safety requirements for a component"""
    
    safety_info = {
        "input_validation": [],
        "output_validation": [],
        "error_handling": [],
        "compliance": []
    }
    
    # Input validation
    input_val = rag.search(
        f"{component} input validation",
        doc_type="api",
        top_k=2
    )
    safety_info["input_validation"] = input_val
    
    # Output validation
    output_val = rag.search(
        f"{component} output validation",
        doc_type="api",
        top_k=2
    )
    safety_info["output_validation"] = output_val
    
    # Error handling
    errors = rag.search(
        f"{component} error handling",
        doc_type="code",
        top_k=2
    )
    safety_info["error_handling"] = errors
    
    # Compliance
    compliance = rag.search(
        f"{component} compliance guardrails",
        top_k=2
    )
    safety_info["compliance"] = compliance
    
    return safety_info

# Use it
safety = get_safety_requirements("BlackScholesModel")
print(f"Input validation checks: {len(safety['input_validation'])}")
print(f"Output validation checks: {len(safety['output_validation'])}")
print(f"Error handling: {len(safety['error_handling'])}")
print(f"Compliance requirements: {len(safety['compliance'])}")
```

### When to Use
- Regulatory compliance checks
- Safety audit preparation
- Validation requirements
- Quality assurance documentation

---

## Pattern 8: Real-Time Monitoring

### Use Case
Monitor and log RAG system performance.

### Implementation

```python
from semai.rag_system import RAGSystem
import time

rag = RAGSystem(".")
rag.build_index()

def monitored_search(query: str, top_k: int = 5) -> dict:
    """Search with performance monitoring"""
    
    start_time = time.time()
    
    # Execute search
    results = rag.search(query, top_k=top_k)
    
    elapsed = time.time() - start_time
    
    # Log results
    stats = {
        "query": query,
        "results_count": len(results),
        "elapsed_time": elapsed,
        "avg_relevance": sum(r['similarity'] for r in results) / len(results) if results else 0,
        "top_result": results[0]['title'] if results else None
    }
    
    # Log for monitoring
    print(f"Query: {query}")
    print(f"  Results: {stats['results_count']}")
    print(f"  Time: {stats['elapsed_time']:.3f}s")
    print(f"  Avg Relevance: {stats['avg_relevance']:.1%}")
    
    return stats

# Monitor searches
queries = ["neural network", "pricing model", "validation"]
metrics = []

for query in queries:
    stats = monitored_search(query)
    metrics.append(stats)

# Analyze patterns
print(f"\nTotal queries: {len(metrics)}")
print(f"Avg response time: {sum(m['elapsed_time'] for m in metrics) / len(metrics):.3f}s")
```

### When to Use
- Performance optimization
- System monitoring
- Bottleneck identification
- Usage analytics

---

## Pattern 9: Multi-Agent Collaboration

### Use Case
Multiple agents working together with shared RAG context.

### Implementation

```python
from crewai import Agent, Task, Crew
from semai.tools.rag_tools import create_rag_tools_for_agents

rag_tools = create_rag_tools_for_agents(".")

# Agent 1: Researcher
researcher = Agent(
    role="Researcher",
    goal="Research and explain derivative pricing concepts",
    tools=rag_tools
)

# Agent 2: Implementer
implementer = Agent(
    role="Implementation Specialist",
    goal="Provide implementation details and code examples",
    tools=rag_tools
)

# Agent 3: Validator
validator = Agent(
    role="Quality Validator",
    goal="Ensure implementations follow safety and validation guidelines",
    tools=rag_tools
)

# Create collaborative tasks
research_task = Task(
    description="Research the Black-Scholes model",
    agent=researcher
)

implementation_task = Task(
    description="Implement Black-Scholes pricing",
    agent=implementer,
    context=[research_task]
)

validation_task = Task(
    description="Validate the implementation for safety and compliance",
    agent=validator,
    context=[implementation_task]
)

# Execute collaboration
crew = Crew(
    agents=[researcher, implementer, validator],
    tasks=[research_task, implementation_task, validation_task]
)

result = crew.kickoff()
```

### When to Use
- Complex projects requiring expertise
- Quality assurance workflows
- Documentation and implementation
- Knowledge transfer between agents

---

## Pattern 10: Feedback Loop

### Use Case
Use search results to improve subsequent searches.

### Implementation

```python
from semai.rag_system import RAGSystem

rag = RAGSystem(".")
rag.build_index()

def iterative_search(initial_query: str, max_iterations: int = 3) -> list:
    """Search with iterative refinement"""
    
    all_results = []
    current_query = initial_query
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}: {current_query}")
        
        # Search
        results = rag.search(current_query, top_k=3)
        all_results.extend(results)
        
        if not results:
            print("  No results found")
            break
        
        # Analyze results for refinement
        top_result = results[0]
        print(f"  Top result: {top_result['title']}")
        
        # Refine query based on top result
        if iteration < max_iterations - 1:
            # Extract keywords from top result for refinement
            keywords = top_result['title'].split()
            if len(keywords) > 1:
                current_query = f"{initial_query} {keywords[0]}"
    
    return all_results

# Use iterative search
results = iterative_search("Black-Scholes")
print(f"\nTotal results collected: {len(results)}")
```

### When to Use
- Complex searches requiring refinement
- Exploratory analysis
- Progressive deepening
- Comprehensive research

---

## Best Practices Summary

1. **Query Formulation**
   - Use specific, descriptive queries
   - Include context and domain terms
   - Avoid single-word searches

2. **Result Validation**
   - Always check relevance scores
   - Verify results match your intent
   - Use multiple search strategies

3. **Performance**
   - Use caching for repeated access
   - Batch search operations
   - Monitor query response times

4. **Integration**
   - Use RAG tools in agents
   - Augment LLM prompts
   - Create context-aware systems

5. **Maintenance**
   - Keep documentation updated
   - Monitor search quality
   - Collect usage metrics

---

## Performance Optimization

### For Fast Searches
```python
# Use simple caching
rag.build_index(use_cache=True)
```

### For Better Results
```python
# Use larger embedding model
rag = RAGSystem(".", embedding_model="all-mpnet-base-v2")
```

### For Large Datasets
```python
# Chunk smaller for more results
loader = DocumentLoader(chunk_size=256)
```

---

## Troubleshooting Integration

### Common Issues

| Issue | Solution |
|-------|----------|
| Poor search results | Use more specific queries |
| Slow initialization | Enable caching |
| Memory errors | Reduce document load or chunk size |
| Missing results | Lower similarity threshold |

---

## Next Steps

1. Review the patterns that match your use case
2. Implement the chosen pattern
3. Monitor performance with metrics
4. Iterate and optimize
5. Document your integration

See [RAG_SYSTEM_GUIDE.md](RAG_SYSTEM_GUIDE.md) for detailed API reference and troubleshooting.
