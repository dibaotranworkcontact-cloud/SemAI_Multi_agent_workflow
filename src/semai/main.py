#!/usr/bin/env python
import sys
import warnings
import os

from datetime import datetime
from dotenv import load_dotenv

from semai.crew import Semai
from semai.validation_crew import ValidationCrew
from semai.data_input_handler import DataExtractionInputInterface, get_data_link_from_user
from semai.human_interface import run_hitl_interface, AgentExecutionMonitor, HumanInterface
from semai.softmax_metrics import get_global_softmax_metrics, TokenSamplingController
from semai.task_feedback_handler import get_task_feedback_handler, reset_feedback_handler
from semai.interactive_crew_executor import execute_crew_with_feedback

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Load environment variables from .env file
load_dotenv()


def check_api_keys():
    """
    Check if required API keys are configured.
    Warns user if keys are missing or still have placeholder values.
    """
    missing_keys = []
    placeholder_patterns = ['your-', 'sk-your-', 'replace-', 'insert-', 'xxx']
    
    # Check OpenAI API Key (required)
    openai_key = os.getenv('OPENAI_API_KEY', '')
    if not openai_key:
        missing_keys.append('OPENAI_API_KEY')
    elif any(pattern in openai_key.lower() for pattern in placeholder_patterns):
        missing_keys.append('OPENAI_API_KEY (still has placeholder value)')
    
    # Check Together API Key (required for DeepSeek models)
    together_key = os.getenv('TOGETHER_API_KEY', '')
    if not together_key:
        missing_keys.append('TOGETHER_API_KEY')
    elif any(pattern in together_key.lower() for pattern in placeholder_patterns):
        missing_keys.append('TOGETHER_API_KEY (still has placeholder value)')
    
    if missing_keys:
        print("\n" + "="*80)
        print("⚠️  WARNING: MISSING OR INVALID API KEYS")
        print("="*80)
        print("\nThe following API keys are missing or have placeholder values:")
        for key in missing_keys:
            print(f"  ❌ {key}")
        print("\n📋 SETUP INSTRUCTIONS:")
        print("  1. Copy .env.example to .env:  cp .env.example .env")
        print("  2. Edit .env and add your actual API keys")
        print("  3. Get API keys from:")
        print("     • OpenAI: https://platform.openai.com/api-keys")
        print("     • Together AI: https://api.together.xyz/settings/api-keys")
        print("="*80)
        
        proceed = input("\nContinue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("❌ Exiting. Please configure your API keys first.")
            sys.exit(1)
        print("")


# Initialize softmax metrics with temperature 0.2 (balanced output, prioritizes high-scoring tokens)
SOFTMAX_TEMPERATURE = 0.2
softmax_metrics = get_global_softmax_metrics(temperature=SOFTMAX_TEMPERATURE)
token_sampler = TokenSamplingController(temperature=SOFTMAX_TEMPERATURE)

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def select_crew():
    """
    Prompt user to select which crew to run.
    Returns the crew type: 'computational' or 'validation'.
    """
    print("\n" + "="*80)
    print("🚀 SELECT CREW TO RUN")
    print("="*80)
    print("  [1] Computational Crew")
    print("      → Data extraction, feature engineering, model training,")
    print("        hyperparameter tuning, and performance evaluation")
    print("")
    print("  [2] Validation Crew")
    print("      → Compliance checking, model replication, robustness testing,")
    print("        risk assessment, and comprehensive documentation")
    print("="*80)
    
    while True:
        choice = input("Select crew (1 or 2): ").strip()
        if choice == '1':
            print("✅ Selected: Computational Crew\n")
            return 'computational'
        elif choice == '2':
            print("✅ Selected: Validation Crew\n")
            return 'validation'
        else:
            print("⚠️  Please enter 1 or 2.")


def get_dataset_path():
    """
    Prompt user to choose data source: local dataset or URL.
    Supports derivative pricing simulated datasets.
    Returns the dataset information and source type.
    """
    from semai.data_input_handler import DataExtractionInputInterface
    
    interface = DataExtractionInputInterface()
    result = interface.interactive_submit()
    
    if not result or result.get('status') != 'success':
        print("❌ Failed to select data source. Exiting.")
        sys.exit(1)
    
    return result


def select_model():
    """
    Prompt user to select an LLM model.
    Returns the selected model name.
    """
    available_models = [
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ]
    
    print("\n" + "="*80)
    print("🤖 SELECT LLM MODEL")
    print("="*80)
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    print("="*80)
    
    while True:
        try:
            choice = input("Enter model number (1-5): ").strip()
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                selected_model = available_models[model_idx]
                print(f"✅ Selected model: {selected_model}\n")
                return selected_model
            else:
                print(f"⚠️  Please enter a number between 1 and {len(available_models)}.")
        except ValueError:
            print("⚠️  Invalid input. Please enter a valid number.")


def get_task_feedback():
    """
    Prompt user for feedback after task execution.
    Returns 'continue', 'end', or feedback text.
    """
    print("\n" + "="*80)
    print("📋 TASK FEEDBACK")
    print("="*80)
    print("Options:")
    print("  [C] Continue to next task")
    print("  [E] End execution")
    print("  [F] Provide feedback")
    print("="*80)
    
    while True:
        choice = input("Enter your choice (C/E/F): ").strip().lower()
        if choice == 'c':
            return 'continue'
        elif choice == 'e':
            return 'end'
        elif choice == 'f':
            feedback = input("Enter your feedback: ").strip()
            if feedback:
                return feedback
            else:
                print("⚠️  Feedback cannot be empty. Please try again.")
        else:
            print("⚠️  Invalid choice. Please enter C, E, or F.")


def run():
    """
    Run the crew with model selection and inter-task feedback.
    Supports both URL and local derivative pricing dataset sources.
    Allows selection between Computational Crew and Validation Crew.
    """
    # Check API keys before starting
    check_api_keys()
    
    # First, let user select which crew to run
    crew_type = select_crew()
    
    if crew_type == 'validation':
        # Run Validation Crew
        run_validation_crew()
        return
    
    # Continue with Computational Crew flow
    # Get dataset source (URL, local file, or predefined dataset)
    dataset_info = get_dataset_path()
    
    # Get model selection from user
    selected_model = select_model()
    
    # Initialize feedback handler
    reset_feedback_handler()
    feedback_handler = get_task_feedback_handler()
    
    # Prepare inputs based on source type
    if dataset_info.get('source_type') == 'local_dataset':
        inputs = {
            'dataset_url': dataset_info.get('file_path'),
            'dataset_type': dataset_info.get('source_type'),
            'dataset_name': dataset_info.get('dataset_name'),
            'subset': dataset_info.get('subset'),
            'features': dataset_info.get('features'),
            'current_year': str(datetime.now().year),
            'selected_model': selected_model,
            'enable_task_feedback': True
        }
    else:
        inputs = {
            'dataset_url': dataset_info.get('file_path') or dataset_info.get('link'),
            'dataset_type': dataset_info.get('source_type'),
            'link_type': dataset_info.get('link_type'),
            'current_year': str(datetime.now().year),
            'selected_model': selected_model,
            'enable_task_feedback': True
        }

    try:
        print("\n" + "="*80)
        print("🚀 STARTING ML PIPELINE CREW")
        print("="*80)
        
        if dataset_info.get('source_type') == 'local_dataset':
            print(f"📊 Dataset: {dataset_info.get('dataset_display_name')}")
            print(f"📋 Subset: {dataset_info.get('subset').upper()}")
            print(f"📁 File: {dataset_info.get('file_path')}")
            print(f"🔢 Features: {', '.join(dataset_info.get('features', []))}")
        else:
            print(f"📊 Dataset Source: {dataset_info.get('file_path') or dataset_info.get('link')}")
            print(f"📋 Source Type: {dataset_info.get('source_type', 'url').upper()}")
        
        print(f"🤖 Model: {selected_model}")
        print(f"📋 Task Feedback: Enabled")
        print("="*80 + "\n")
        
        crew_instance = Semai()
        # Update all agents with selected model
        crew_instance.update_agent_models(selected_model)
        
        crew = crew_instance.crew()
        
        # Execute with interactive feedback (pass feedback_handler separately)
        result = execute_crew_with_feedback(crew, inputs, feedback_handler)
        
        print("\n" + "="*80)
        print("✅ ML PIPELINE EXECUTION COMPLETED!")
        print("="*80)
        
        # Display execution summary
        if isinstance(result, dict):
            print(f"\n📊 EXECUTION SUMMARY:")
            print(f"  • Tasks Completed: {result.get('tasks_completed', 0)}")
            print(f"  • Execution Halted: {'Yes' if result.get('execution_halted') else 'No'}")
            
            # Display collected feedbacks
            all_feedbacks = result.get('feedbacks', {})
            if all_feedbacks:
                print(f"\n📝 TASK FEEDBACKS COLLECTED:")
                print("-" * 80)
                for task_name, feedback in all_feedbacks.items():
                    print(f"  • {task_name}: {feedback}")
                print("-" * 80)
        
        print("\n")
        
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def run_validation_crew():
    """
    Run the Validation Crew for model risk management and compliance.
    Requires path to Computational Crew documentation output.
    """
    print("\n" + "="*80)
    print("🔍 VALIDATION CREW - Model Risk Management")
    print("="*80)
    
    # Get path to Computational Crew documentation
    print("\nThe Validation Crew requires the documentation output from")
    print("a previous Computational Crew run.")
    print("")
    
    while True:
        doc_path = input("Enter path to Computational Crew documentation folder: ").strip()
        
        if not doc_path:
            print("⚠️  Path cannot be empty. Please try again.")
            continue
            
        if os.path.exists(doc_path):
            break
        else:
            print(f"⚠️  Path not found: {doc_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("❌ Validation cancelled.")
                return
    
    # Initialize feedback handler
    reset_feedback_handler()
    feedback_handler = get_task_feedback_handler()
    
    # Prepare inputs for Validation Crew
    inputs = {
        'computational_crew_output_path': doc_path,
        'current_year': str(datetime.now().year),
        'enable_task_feedback': True
    }
    
    try:
        print("\n" + "="*80)
        print("🚀 STARTING VALIDATION CREW")
        print("="*80)
        print(f"📁 Documentation Path: {doc_path}")
        print(f"📋 Task Feedback: Enabled")
        print("="*80 + "\n")
        
        crew_instance = ValidationCrew()
        crew = crew_instance.crew()
        
        # Execute with interactive feedback
        result = execute_crew_with_feedback(crew, inputs, feedback_handler)
        
        print("\n" + "="*80)
        print("✅ VALIDATION CREW EXECUTION COMPLETED!")
        print("="*80)
        
        # Display execution summary
        if isinstance(result, dict):
            print(f"\n📊 EXECUTION SUMMARY:")
            print(f"  • Tasks Completed: {result.get('tasks_completed', 0)}")
            print(f"  • Execution Halted: {'Yes' if result.get('execution_halted') else 'No'}")
            
            # Display collected feedbacks
            all_feedbacks = result.get('feedbacks', {})
            if all_feedbacks:
                print(f"\n📝 TASK FEEDBACKS COLLECTED:")
                print("-" * 80)
                for task_name, feedback in all_feedbacks.items():
                    print(f"  • {task_name}: {feedback}")
                print("-" * 80)
        
        print("\n")
        
    except Exception as e:
        raise Exception(f"An error occurred while running the Validation Crew: {e}")


def run_with_human_feedback():
    """
    Run the crew with Human-in-the-Loop interface.
    Allows user to select agents, provide feedback, and monitor execution.
    Uses softmax metrics (T=0.2) for balanced output generation.
    """
    # Run the HITL interface
    session_config = run_hitl_interface()
    
    if not session_config:
        print("❌ Session configuration failed. Exiting.")
        sys.exit(1)
    
    # Extract configuration
    agent_id = session_config.get('agent_id')
    human_feedback = session_config.get('human_feedback')
    additional_instructions = session_config.get('additional_instructions', [])
    
    # Prepare inputs for crew
    inputs = {
        'task_description': session_config.get('task_description'),
        'human_feedback': human_feedback,
        'additional_instructions': '\n'.join(additional_instructions) if additional_instructions else '',
        'current_year': str(datetime.now().year),
        'hitl_enabled': True,
        'softmax_temperature': SOFTMAX_TEMPERATURE
    }
    
    # Create execution monitor
    interface = HumanInterface()
    monitor = AgentExecutionMonitor(interface)
    
    try:
        print("\n" + "="*80)
        print("🚀 STARTING CREW EXECUTION WITH HUMAN FEEDBACK")
        print(f"📊 Softmax Metrics Enabled (Temperature: {SOFTMAX_TEMPERATURE} - High-scoring tokens prioritized)")
        print("="*80 + "\n")
        
        monitor.display_execution_start(session_config.get('agent_selected'))
        
        # Run crew
        result = Semai().crew().kickoff(inputs=inputs)
        
        monitor.display_execution_complete(
            session_config.get('agent_selected'),
            str(result) if result else "Task completed successfully"
        )
        
        # Display softmax metrics summary
        metrics_summary = softmax_metrics.get_metrics_summary()
        if metrics_summary:
            print("\n" + "="*80)
            print("📊 SOFTMAX METRICS SUMMARY")
            print("="*80)
            print(f"Average Entropy (Variability): {metrics_summary.get('avg_entropy', 0):.4f}")
            print(f"Average Confidence (Determinism): {metrics_summary.get('avg_confidence', 0):.4f}")
            print(f"Average Gini (Distribution Concentration): {metrics_summary.get('avg_gini', 0):.4f}")
            print(f"Total Predictions: {metrics_summary.get('total_predictions', 0)}")
            print("="*80 + "\n")
        
        # Ask if user wants to continue with more agents
        print("="*80)
        while True:
            response = input("🔄 Do you want to continue with another agent? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                run_with_human_feedback()
                break
            elif response in ['no', 'n']:
                print("✅ All tasks completed. Thank you for using SEMAI!")
                break
            else:
                print("⚠️  Please enter 'yes' or 'no'.\n")
    
    except Exception as e:
        monitor.display_execution_error(
            session_config.get('agent_selected'),
            str(e)
        )
        raise


def run_with_data_link(data_link: str = None):
    """
    Run the crew with user-provided data link for extraction.
    Uses softmax metrics (T=0.2) for deterministic, high-confidence outputs.
    
    Args:
        data_link: Optional data link. If not provided, user will be prompted interactively.
    """
    interface = DataExtractionInputInterface()
    
    # Get data link either from parameter or interactively
    if data_link:
        result = interface.programmatic_submit(data_link)
    else:
        result = interface.interactive_submit()
    
    if result and result["status"] == "success":
        # Get the context to pass to crew
        crew_context = interface.get_crew_context()
        
        inputs = {
            'data_link': crew_context['data_link'],
            'link_type': crew_context['link_type'],
            'instruction': crew_context['instruction'],
            'current_year': str(datetime.now().year),
            'softmax_temperature': SOFTMAX_TEMPERATURE
        }
        
        try:
            print("\n📊 Starting ML Pipeline with provided data link...")
            print(f"📍 Data Source: {result['link_type'].upper()}")
            print(f"🔗 Link: {result['link']}")
            print(f"🌡️  Softmax Temperature: {SOFTMAX_TEMPERATURE} (High-scoring tokens prioritized)\n")
            
            Semai().crew().kickoff(inputs=inputs)
            
            # Display metrics
            metrics_summary = softmax_metrics.get_metrics_summary()
            if metrics_summary:
                print("\n📊 Execution Metrics:")
                print(f"   Avg Confidence: {metrics_summary.get('avg_confidence', 0):.4f}")
                print(f"   Total Tokens Generated: {metrics_summary.get('total_predictions', 0)}")
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")
    else:
        print("\n❌ Cannot proceed without a valid data link. Exiting.")
        sys.exit(1)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        Semai().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Semai().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        Semai().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = Semai().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
