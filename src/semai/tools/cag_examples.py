"""
Corrective Augmented Generation (CAG) Tool - Usage Examples

This module provides 6 complete working examples demonstrating how to use
the CAGTool for content validation and improvement.

Examples:
1. Basic Usage: Simple content validation and improvement
2. Reference-Based Validation: Comparing content with reference material
3. Batch Processing: Processing multiple content items
4. Custom Validation Criteria: Focusing on specific quality aspects
5. Iterative Refinement: Multi-stage improvement workflow
6. CrewAI Integration: Using CAG within a crew agent
"""

from semai.tools.corrective_augmented_generation import (
    CAGTool,
    CorrectivenessValidator,
    AugmentationEngine,
    AugmentationStrategy,
    ValidationCriteria
)


# ============================================================================
# Example 1: Basic Usage
# ============================================================================
def example_1_basic_usage():
    """
    Example 1: Basic usage of CAGTool.
    
    Demonstrates simple validation and automatic improvement of content.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic CAGTool Usage")
    print("="*70)
    
    # Initialize CAG tool
    cag = CAGTool(
        max_iterations=3,
        quality_threshold=0.75,
        auto_improve=True,
        verbose=True
    )
    
    # Sample content with some issues
    content = """
    The derivative pricing is important. We use models to price derivatives.
    The models is very complex. We need to understand the models. The Greeks
    helps us understand risk. Delta measures how much the price changes when
    the underlying price changes. Gamma measures the rate of change of Delta.
    Vega measures the sensitivity to volatility. Rho measures the sensitivity
    to interest rates. Theta measures the time decay of the option value.
    """
    
    # Run CAG process
    result = cag._run(content, auto_improve=True, verbose=True)
    
    # Display results
    print("\n--- Results ---")
    print(f"Original length: {len(content)} characters")
    print(f"Final length: {len(result['final_content'])} characters")
    print(f"Quality score: {result['quality_score']:.2%}")
    print(f"Iterations used: {result['iterations_used']}")
    print(f"\nValidation scores:")
    for criterion, score in result['validation_scores'].items():
        print(f"  {criterion}: {score:.2%}")
    
    print(f"\nImproved content:\n{result['final_content']}")
    
    return result


# ============================================================================
# Example 2: Reference-Based Validation
# ============================================================================
def example_2_reference_validation():
    """
    Example 2: Validation with reference content.
    
    Demonstrates comparing generated content against reference material.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Reference-Based Validation")
    print("="*70)
    
    cag = CAGTool(max_iterations=2, quality_threshold=0.80, verbose=True)
    
    # Reference content
    reference = """
    Black-Scholes model for option pricing assumes several things.
    The stock price follows geometric Brownian motion. There are no
    arbitrage opportunities. Trading is continuous. There are no
    transaction costs. The risk-free rate is constant. Volatility
    is constant. The option is European style.
    """
    
    # Generated content (should relate to reference)
    generated = """
    The Black-Scholes model is used for pricing options. It assumes
    the stock follows a random walk model. The model uses constant
    volatility and interest rates. Arbitrage is not possible in
    this framework. European options can be priced exactly.
    """
    
    # Validate against reference
    result = cag._run(
        generated,
        reference=reference,
        auto_improve=True,
        verbose=True
    )
    
    print("\n--- Results ---")
    print(f"Relevance to reference: {result['validation_scores']['relevance']:.2%}")
    print(f"Overall quality: {result['quality_score']:.2%}")
    print(f"Reference was used: {result['reference_used']}")
    
    print(f"\nGenerated content:")
    print(generated)
    print(f"\nImproved content:")
    print(result['final_content'])
    
    return result


# ============================================================================
# Example 3: Batch Processing
# ============================================================================
def example_3_batch_processing():
    """
    Example 3: Processing multiple content items in batch.
    
    Demonstrates how to validate and improve multiple pieces of content.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing Multiple Content Items")
    print("="*70)
    
    cag = CAGTool(max_iterations=2, quality_threshold=0.70, verbose=False)
    
    # Multiple content items to process
    content_items = [
        "Machine learning is the future. It is used everywhere. Algorithms learns from data.",
        "Neural networks are inspired by the brain. They have neurons and connections.",
        "Deep learning uses many layers of neural networks for learning representations.",
    ]
    
    batch_results = []
    
    for i, content in enumerate(content_items, 1):
        print(f"\nProcessing item {i}...")
        result = cag._run(content, auto_improve=True)
        batch_results.append(result)
        print(f"  Quality: {result['quality_score']:.2%}")
        print(f"  Changes: {len(result['changes_made'])} improvements")
    
    # Summary statistics
    print("\n--- Batch Summary ---")
    avg_quality = sum(r['quality_score'] for r in batch_results) / len(batch_results)
    print(f"Average quality improvement: {avg_quality:.2%}")
    print(f"Total items processed: {len(batch_results)}")
    
    return batch_results


# ============================================================================
# Example 4: Custom Validation Criteria
# ============================================================================
def example_4_custom_criteria():
    """
    Example 4: Using individual validators for focused assessment.
    
    Demonstrates using CorrectivenessValidator directly with specific criteria.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Validation Criteria")
    print("="*70)
    
    validator = CorrectivenessValidator(verbose=True)
    
    content = """
    Hyperparameter tuning is very very important for machine learning models.
    We needs to find the best hyperparameters. The hyperparameters controls
    how the model trains. Random search and grid search is methods for tuning.
    Random search utilizes random sampling of the hyperparameter space.
    Grid search uses a predefined grid of values to search through.
    """
    
    print("\nValidating content with custom criteria...\n")
    
    # Run individual validators
    accuracy = validator.validate_accuracy(content)
    completeness = validator.validate_completeness(content)
    clarity = validator.validate_clarity(content)
    consistency = validator.validate_consistency(content)
    
    print("\n--- Validation Results ---")
    print(f"Accuracy: {accuracy.score:.2%}")
    print(f"  Issues: {accuracy.issues}")
    print(f"\nCompleteness: {completeness.score:.2%}")
    print(f"  Issues: {completeness.issues}")
    print(f"\nClarity: {clarity.score:.2%}")
    print(f"  Issues: {clarity.issues}")
    print(f"\nConsistency: {consistency.score:.2%}")
    print(f"  Issues: {consistency.issues}")
    
    return {
        'accuracy': accuracy,
        'completeness': completeness,
        'clarity': clarity,
        'consistency': consistency
    }


# ============================================================================
# Example 5: Iterative Refinement Workflow
# ============================================================================
def example_5_iterative_refinement():
    """
    Example 5: Multi-stage refinement workflow.
    
    Demonstrates progressively improving content through multiple strategies.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Iterative Refinement Workflow")
    print("="*70)
    
    augmenter = AugmentationEngine(verbose=True)
    
    content = "ML is good. It works well. We use it for predictions."
    
    print(f"Original: {content}\n")
    
    # Stage 1: Expand
    print("Stage 1: Expanding content...")
    expanded = augmenter.expand(content, expansion_ratio=2.0)
    current = expanded.improved_content
    print(f"Result: {current}\n")
    
    # Stage 2: Clarify
    print("Stage 2: Clarifying content...")
    clarified = augmenter.clarify(current)
    current = clarified.improved_content
    print(f"Result: {current}\n")
    
    # Stage 3: Correct
    print("Stage 3: Correcting errors...")
    corrected = augmenter.correct(current)
    current = corrected.improved_content
    print(f"Result: {current}\n")
    
    # Summary
    print("--- Refinement Summary ---")
    total_changes = (
        len(expanded.changes_made) +
        len(clarified.changes_made) +
        len(corrected.changes_made)
    )
    print(f"Total improvements: {total_changes}")
    print(f"Original length: {len(content)} chars")
    print(f"Final length: {len(current)} chars")
    
    return current


# ============================================================================
# Example 6: CrewAI Integration
# ============================================================================
def example_6_crewai_integration():
    """
    Example 6: Integration with CrewAI agents.
    
    Demonstrates using CAGTool as a task output processor in a crew.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: CrewAI Integration Concept")
    print("="*70)
    
    # In actual CrewAI usage, you would:
    # 1. Create a tool instance
    # 2. Register it with an agent
    # 3. Use it to validate outputs from other agents
    
    cag_tool = CAGTool(
        max_iterations=3,
        quality_threshold=0.80,
        auto_improve=True
    )
    
    print("""
    Integration Pattern:
    
    1. Create CAGTool instance
    2. Register with ValidationAgent:
        @agent
        def validation_agent():
            return Agent(
                role="Quality Validator",
                goal="Ensure all outputs meet quality standards",
                tools=[cag_tool],
                llm=...
            )
    
    3. Use in ValidationTask:
        @task
        def validation_task():
            return Task(
                description="Validate and improve agent outputs",
                agent=validation_agent(),
                tools=[cag_tool]
            )
    
    4. Run validation in crew:
        result = cag_tool._run(
            content=agent_output,
            auto_improve=True
        )
    """)
    
    # Example: Process sample agent output
    sample_agent_output = """
    The analysis shows that derivative pricing models requires several key inputs.
    The inputs includes the current price, strike price, time to expiration, 
    volatility, and interest rates. The most common model are the Black-Scholes 
    model and the binomial model. Traders uses these models for making trading 
    decisions. The accuracy of the model depends on the accuracy of the inputs.
    """
    
    print("Sample agent output:")
    print(sample_agent_output)
    
    result = cag_tool._run(sample_agent_output, auto_improve=True, verbose=False)
    
    print("\nValidated and improved output:")
    print(result['final_content'])
    print(f"\nQuality improvement: {result['quality_score']:.2%}")
    
    return result


# ============================================================================
# Main Execution
# ============================================================================
def run_all_examples():
    """Run all CAG tool examples."""
    
    print("\n" + "="*70)
    print("CORRECTIVE AUGMENTED GENERATION (CAG) TOOL - EXAMPLES")
    print("="*70)
    
    try:
        example_1_basic_usage()
        example_2_reference_validation()
        example_3_batch_processing()
        example_4_custom_criteria()
        example_5_iterative_refinement()
        example_6_crewai_integration()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
