"""
Corrective Augmented Generation (CAG) Tool - CrewAI Integration Patterns

This module demonstrates 4 integration patterns for using the CAGTool
within CrewAI crews and agents.

Integration Patterns:
1. Documentation Validation Agent: Validates all generated documentation
2. Evaluation Agent: Comprehensive output evaluation and improvement
3. CAG Task: Dedicated validation task in a crew
4. Quality Assurance Crew: Multi-agent quality workflow
"""

from typing import Any, Dict, List, Optional
from crewai import Agent, Task, Crew
from semai.tools.corrective_augmented_generation import (
    CAGTool,
    CorrectivenessValidator,
    AugmentationEngine
)


# ============================================================================
# Pattern 1: Documentation Validation Agent
# ============================================================================
class DocumentationValidationAgent:
    """
    Agent specialized in validating and improving generated documentation.
    
    This agent can be used to ensure all documentation outputs meet
    quality standards before publication.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the documentation validation agent.
        
        Args:
            llm_config: LLM configuration dictionary
        """
        self.cag_tool = CAGTool(
            max_iterations=3,
            quality_threshold=0.80,
            auto_improve=True,
            verbose=False
        )
        self.llm_config = llm_config or {}
    
    def create_agent(self) -> Agent:
        """
        Create a CrewAI Agent for documentation validation.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Documentation Validator",
            goal="Ensure all documentation is accurate, complete, and clear",
            backstory="""You are an expert technical writer and editor.
            Your role is to validate and improve documentation to ensure
            it meets high quality standards for accuracy, completeness,
            clarity, consistency, and relevance.""",
            tools=[self.cag_tool],
            verbose=True,
            # Note: In actual implementation, add llm config
            # llm=create_llm_from_config(self.llm_config)
        )
        return agent
    
    def validate_documentation(self, content: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate documentation content.
        
        Args:
            content: Documentation to validate
            reference: Optional reference documentation
            
        Returns:
            Validation results
        """
        result = self.cag_tool._run(
            content=content,
            reference=reference,
            auto_improve=True,
            verbose=False
        )
        return result


# ============================================================================
# Pattern 2: Comprehensive Evaluation Agent
# ============================================================================
class ComprehensiveEvaluationAgent:
    """
    Agent that performs comprehensive evaluation and improvement of content.
    
    Combines validation and augmentation for thorough quality assessment.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation agent.
        
        Args:
            llm_config: LLM configuration dictionary
        """
        self.validator = CorrectivenessValidator(verbose=False)
        self.augmenter = AugmentationEngine(verbose=False)
        self.cag_tool = CAGTool(
            max_iterations=4,
            quality_threshold=0.85,
            auto_improve=True,
            verbose=False
        )
        self.llm_config = llm_config or {}
    
    def create_agent(self) -> Agent:
        """
        Create a CrewAI Agent for comprehensive evaluation.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Quality Assurance Specialist",
            goal="Comprehensively evaluate and improve content quality",
            backstory="""You are a senior quality assurance specialist.
            Your expertise is in identifying and correcting quality issues
            across multiple dimensions including accuracy, completeness,
            clarity, consistency, and relevance. You use advanced validation
            and augmentation techniques to ensure exceptional output quality.""",
            tools=[self.cag_tool],
            verbose=True,
            # Note: Add llm config in actual implementation
        )
        return agent
    
    def perform_evaluation(
        self,
        content: str,
        evaluation_type: str = "comprehensive",
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of content.
        
        Args:
            content: Content to evaluate
            evaluation_type: Type of evaluation (comprehensive, strict, lenient)
            reference: Optional reference content
            
        Returns:
            Comprehensive evaluation results
        """
        # Adjust thresholds based on evaluation type
        thresholds = {
            "comprehensive": 0.80,
            "strict": 0.90,
            "lenient": 0.70
        }
        
        threshold = thresholds.get(evaluation_type, 0.80)
        
        self.cag_tool.quality_threshold = threshold
        
        result = self.cag_tool._run(
            content=content,
            reference=reference,
            auto_improve=True,
            verbose=False
        )
        
        # Add evaluation metadata
        result['evaluation_type'] = evaluation_type
        result['threshold_used'] = threshold
        
        return result


# ============================================================================
# Pattern 3: CAG Validation Task
# ============================================================================
class CAGValidationTask:
    """
    Creates a CrewAI Task for content validation using CAGTool.
    
    This pattern can be used in any crew to add validation functionality.
    """
    
    @staticmethod
    def create_validation_task(
        agent: Agent,
        content_from_task: Optional[str] = None,
        reference: Optional[str] = None
    ) -> Task:
        """
        Create a validation task.
        
        Args:
            agent: Agent to execute the task
            content_from_task: Optional content from previous task
            reference: Optional reference material
            
        Returns:
            Configured Task instance
        """
        cag_tool = CAGTool(
            max_iterations=3,
            quality_threshold=0.75,
            auto_improve=True
        )
        
        task_description = """
        Validate and improve the provided content using comprehensive quality criteria:
        1. Assess accuracy against any provided reference
        2. Ensure completeness of content
        3. Improve clarity and readability
        4. Verify internal consistency
        5. Confirm relevance to the topic
        
        Provide a validated version with quality scores for each criterion.
        """
        
        if reference:
            task_description += f"\n\nReference material for comparison:\n{reference}"
        
        task = Task(
            description=task_description,
            agent=agent,
            expected_output="""
            Validated content with:
            - Quality score for each criterion
            - List of improvements made
            - Final improved content
            - Recommendations for further improvement
            """,
            tools=[cag_tool],
            input=content_from_task or "Content for validation"
        )
        
        return task


# ============================================================================
# Pattern 4: Quality Assurance Crew
# ============================================================================
class QualityAssuranceCrew:
    """
    A multi-agent crew specialized in quality assurance and content improvement.
    
    This crew uses multiple agents with different expertise for thorough
    quality validation and improvement.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the QA crew.
        
        Args:
            llm_config: LLM configuration dictionary
        """
        self.llm_config = llm_config or {}
        self.cag_tool = CAGTool(
            max_iterations=3,
            quality_threshold=0.80,
            auto_improve=True
        )
    
    def create_crew(self) -> Crew:
        """
        Create a Quality Assurance Crew.
        
        Returns:
            Configured Crew instance
        """
        # Validation Agent
        validator_agent = Agent(
            role="Content Validator",
            goal="Identify quality issues and validation errors",
            backstory="""You are an expert in content validation.
            You specialize in identifying accuracy issues, incomplete
            information, clarity problems, and consistency errors.""",
            tools=[self.cag_tool],
            verbose=False,
            # Add llm config
        )
        
        # Improvement Agent
        improver_agent = Agent(
            role="Content Improver",
            goal="Enhance and refine content quality",
            backstory="""You are a master editor and content improver.
            You excel at expanding content, clarifying complex concepts,
            simplifying verbose text, and correcting errors.""",
            tools=[self.cag_tool],
            verbose=False,
            # Add llm config
        )
        
        # Review Agent
        reviewer_agent = Agent(
            role="Quality Reviewer",
            goal="Perform final quality assurance review",
            backstory="""You are a senior quality reviewer.
            You conduct comprehensive final reviews to ensure all
            quality standards have been met before publication.""",
            tools=[self.cag_tool],
            verbose=False,
            # Add llm config
        )
        
        # Validation Task
        validation_task = Task(
            description="""Validate the provided content across all quality criteria.
            Identify and document any issues found.""",
            agent=validator_agent,
            expected_output="Detailed validation report with issues identified"
        )
        
        # Improvement Task
        improvement_task = Task(
            description="""Improve the content based on validation findings.
            Apply appropriate augmentation strategies to enhance quality.""",
            agent=improver_agent,
            expected_output="Improved content with change log",
            context=[validation_task]
        )
        
        # Review Task
        review_task = Task(
            description="""Perform final quality review of the improved content.
            Ensure all quality standards are met.""",
            agent=reviewer_agent,
            expected_output="Final review report with quality metrics",
            context=[improvement_task]
        )
        
        # Create Crew
        crew = Crew(
            agents=[validator_agent, improver_agent, reviewer_agent],
            tasks=[validation_task, improvement_task, review_task],
            verbose=True
        )
        
        return crew


# ============================================================================
# Usage Examples
# ============================================================================
def example_documentation_validation():
    """Example usage of Documentation Validation Agent."""
    
    print("\n" + "="*70)
    print("Pattern 1: Documentation Validation Agent")
    print("="*70)
    
    validator = DocumentationValidationAgent()
    agent = validator.create_agent()
    
    sample_doc = """
    The Black-Scholes model is used for option pricing. The model assumes
    several things: geometric brownian motion, no arbitrage, continuous
    trading. The formula is complex. The model is widely used. It have
    limitations. Traders uses it for hedging.
    """
    
    print(f"\nValidating documentation:\n{sample_doc}")
    
    result = validator.validate_documentation(sample_doc)
    
    print(f"\nQuality Score: {result['quality_score']:.2%}")
    print(f"Changes Made: {len(result['changes_made'])}")
    
    return result


def example_comprehensive_evaluation():
    """Example usage of Comprehensive Evaluation Agent."""
    
    print("\n" + "="*70)
    print("Pattern 2: Comprehensive Evaluation Agent")
    print("="*70)
    
    evaluator = ComprehensiveEvaluationAgent()
    
    sample_content = """
    Machine learning models requires training data. The data should be
    representative of the problem domain. Data preprocessing is very important.
    We need to handles missing values and outliers. Feature engineering is
    the process of creating new features. The features should be relevant
    to the prediction task. Model selection is critical.
    """
    
    print(f"\nEvaluating content (comprehensive):\n{sample_content}")
    
    result = evaluator.perform_evaluation(sample_content, evaluation_type="comprehensive")
    
    print(f"\nQuality Score: {result['quality_score']:.2%}")
    print(f"Threshold Used: {result['threshold_used']:.2%}")
    
    return result


def example_qa_crew():
    """Example usage of Quality Assurance Crew."""
    
    print("\n" + "="*70)
    print("Pattern 4: Quality Assurance Crew Concept")
    print("="*70)
    
    print("""
    QualityAssuranceCrew creates a multi-agent workflow:
    
    1. Content Validator Agent
       - Analyzes content across 5 quality criteria
       - Identifies issues and problems
       
    2. Content Improver Agent
       - Applies targeted augmentation strategies
       - Refines and enhances the content
       
    3. Quality Reviewer Agent
       - Performs final quality assurance
       - Ensures standards are met
    
    Workflow:
    Input Content → Validation → Improvement → Review → Output
    
    Each agent uses the CAGTool to perform specialized functions
    in the overall quality assurance pipeline.
    """)
    
    qa_crew = QualityAssuranceCrew()
    crew = qa_crew.create_crew()
    
    print("Quality Assurance Crew created successfully!")
    print(f"Crew has {len(crew.agents)} agents and {len(crew.tasks)} tasks")
    
    return crew


def run_all_patterns():
    """Run all integration pattern examples."""
    
    print("\n" + "="*70)
    print("CAG TOOL - CREWAI INTEGRATION PATTERNS")
    print("="*70)
    
    try:
        example_documentation_validation()
        example_comprehensive_evaluation()
        example_qa_crew()
        
        print("\n" + "="*70)
        print("All integration patterns demonstrated successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running patterns: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_patterns()
