"""
Human Interface for Agent Selection and Task Execution
Implements Human-in-the-Loop (HITL) interface for the crew
"""

from typing import Optional, Dict, List, Any
import sys
from enum import Enum


class AgentOption(Enum):
    """Available agent options"""
    DATA_EXTRACTION = ("1", "Data Extraction Agent", "data_extraction_agent")
    EDA = ("2", "EDA Agent", "eda_agent")
    FEATURE_ENGINEERING = ("3", "Feature Engineering Agent", "feature_engineering_agent")
    META_TUNING = ("4", "Meta-Tuning Agent", "meta_tuning_agent")
    MODEL_TRAINING = ("5", "Model Training Agent", "model_training_agent")
    MODEL_EVALUATION = ("6", "Model Evaluation Agent", "model_evaluation_agent")
    
    def __init__(self, number, display_name, agent_id):
        self.number = number
        self.display_name = display_name
        self.agent_id = agent_id


class TaskContext(Enum):
    """Task contexts for different agent selections"""
    EXTRACTION = "Extract data from external source and split into train and test set"
    EDA = "Perform exploratory data analysis on the dataset"
    FEATURE_ENGINEERING = "Engineer features and build preprocessing pipeline"
    META_TUNING = "Perform hyperparameter tuning and model selection"
    TRAINING = "Train the selected model with optimal parameters"
    EVALUATION = "Evaluate the trained model on test data"


class HumanInterface:
    """Interactive human interface for agent selection and task execution"""
    
    def __init__(self):
        self.selected_agent: Optional[AgentOption] = None
        self.agent_description: Optional[str] = None
        self.human_feedback: Optional[str] = None
        self.additional_instructions: List[str] = []
        self.execution_history: List[Dict[str, Any]] = []
    
    def display_header(self):
        """Display interface header"""
        print("\n" + "="*80)
        print(" "*20 + "🤖 SEMAI - ML PIPELINE WITH HUMAN FEEDBACK 🤖")
        print("="*80 + "\n")
    
    def display_task_description(self, task: str):
        """Display task description"""
        print("📋 TASK DESCRIPTION:")
        print(f"   {task}\n")
    
    def display_agent_selection_menu(self):
        """Display agent selection menu"""
        print("👤 SELECT THE AGENT YOU WANT TO USE FOR THIS ITERATION:\n")
        
        for agent in AgentOption:
            print(f"   {agent.number}. {agent.display_name}")
        
        print()
    
    def get_agent_selection(self) -> Optional[AgentOption]:
        """Get user selection for agent"""
        self.display_agent_selection_menu()
        
        while True:
            try:
                user_input = input("   📌 Enter the associated Agent number: ").strip()
                
                for agent in AgentOption:
                    if agent.number == user_input:
                        self.selected_agent = agent
                        return agent
                
                print(f"   ❌ Invalid selection. Please enter a number between 1 and 6.\n")
            
            except KeyboardInterrupt:
                print("\n   ⚠️  Operation cancelled by user.")
                return None
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    def display_selected_agent(self):
        """Display selected agent information"""
        if self.selected_agent:
            print(f"\n✅ WORKING AGENT: {self.selected_agent.display_name}")
            print(f"   Agent ID: {self.selected_agent.agent_id}\n")
    
    def display_task_initialization(self, agent: AgentOption):
        """Display task initialization message"""
        task_map = {
            AgentOption.DATA_EXTRACTION: "Extract data from Kaggle/external source, split into train/test sets",
            AgentOption.EDA: "Perform in-depth exploratory data analysis",
            AgentOption.FEATURE_ENGINEERING: "Engineer features and build preprocessing pipeline",
            AgentOption.META_TUNING: "Conduct hyperparameter tuning and model selection",
            AgentOption.MODEL_TRAINING: "Train the selected model with optimal parameters",
            AgentOption.MODEL_EVALUATION: "Evaluate trained model on test data"
        }
        
        task_desc = task_map.get(agent, "Execute assigned task")
        print(f"🚀 STARTING TASK: {task_desc}\n")
    
    def get_human_feedback(self) -> str:
        """Get human feedback for the task"""
        print("💬 HUMAN FEEDBACK:")
        print("   Provide any specific instructions or feedback for this task.")
        print("   (Leave empty for default execution)\n")
        
        feedback = input("   📝 Your feedback: ").strip()
        self.human_feedback = feedback if feedback else "Proceed with default execution"
        
        return self.human_feedback
    
    def display_feedback_summary(self):
        """Display feedback summary"""
        print(f"\n📋 FEEDBACK SUMMARY:")
        print(f"   {self.human_feedback}\n")
    
    def display_action_execution(self, action_type: str, action_input: str):
        """Display action execution details"""
        print("⚙️  ACTION EXECUTION:")
        print(f"   Action: {action_type}")
        print(f"   Input: {action_input}\n")
    
    def display_final_answer(self, answer: str):
        """Display final answer from agent"""
        print("✨ FINAL ANSWER:")
        print(f"   {answer}\n")
    
    def get_additional_instructions(self) -> bool:
        """Get additional instructions from user"""
        print("💭 HUMAN INTERFACE:")
        
        while True:
            instruction = input("   Provide additional instruction to the Agent (type 'end' to stop): ").strip()
            
            if instruction.lower() == 'end':
                print("\n   ✅ Task execution completed.\n")
                return False
            elif instruction:
                self.additional_instructions.append(instruction)
                print(f"   ✓ Instruction recorded: '{instruction}'")
                print()
                return True
            else:
                print("   ⚠️  Please enter a valid instruction or type 'end' to stop.\n")
    
    def display_execution_history(self):
        """Display execution history"""
        if not self.execution_history:
            return
        
        print("\n📜 EXECUTION HISTORY:")
        print("="*80)
        
        for idx, entry in enumerate(self.execution_history, 1):
            print(f"\n{idx}. Agent: {entry['agent']}")
            print(f"   Task: {entry['task']}")
            print(f"   Feedback: {entry['feedback']}")
            print(f"   Status: {entry['status']}")
    
    def log_execution(self, agent: str, task: str, feedback: str, status: str):
        """Log execution details"""
        self.execution_history.append({
            "agent": agent,
            "task": task,
            "feedback": feedback,
            "status": status
        })
    
    def run_interactive_session(self) -> Dict[str, Any]:
        """
        Run interactive human-in-the-loop session
        
        Returns:
            dict: Session configuration and inputs for crew execution
        """
        self.display_header()
        self.display_task_description(
            "Extract data from external source and split into train and test set"
        )
        
        # Get agent selection
        agent = self.get_agent_selection()
        if not agent:
            print("❌ No agent selected. Exiting.")
            return {}
        
        self.display_selected_agent()
        self.display_task_initialization(agent)
        
        # Get human feedback
        feedback = self.get_human_feedback()
        self.display_feedback_summary()
        
        # Get additional instructions
        instructions = []
        while True:
            if not self.get_additional_instructions():
                break
            instructions.extend(self.additional_instructions)
        
        # Prepare session configuration
        session_config = {
            "agent_selected": agent.display_name,
            "agent_id": agent.agent_id,
            "agent_number": agent.number,
            "task_description": "Extract data from external source and split into train and test set",
            "human_feedback": feedback,
            "additional_instructions": instructions,
            "hitl_enabled": True
        }
        
        # Log execution
        self.log_execution(
            agent=agent.display_name,
            task="Extract data from external source and split into train and test set",
            feedback=feedback,
            status="Pending"
        )
        
        return session_config


class AgentExecutionMonitor:
    """Monitor and display agent execution"""
    
    def __init__(self, interface: HumanInterface):
        self.interface = interface
    
    def display_execution_start(self, agent: str):
        """Display execution start message"""
        print(f"\n🔄 Executing {agent}...")
        print("-" * 80)
    
    def display_execution_progress(self, message: str):
        """Display execution progress"""
        print(f"   ⏳ {message}")
    
    def display_execution_complete(self, agent: str, result: str):
        """Display execution completion"""
        print("-" * 80)
        print(f"✅ {agent} execution completed successfully.\n")
        
        self.interface.display_final_answer(result)
    
    def display_execution_error(self, agent: str, error: str):
        """Display execution error"""
        print("-" * 80)
        print(f"❌ {agent} execution failed.")
        print(f"   Error: {error}\n")


def run_hitl_interface() -> Dict[str, Any]:
    """
    Run the Human-in-the-Loop interface
    
    Returns:
        dict: Session configuration for crew execution
    """
    interface = HumanInterface()
    return interface.run_interactive_session()


if __name__ == "__main__":
    session_config = run_hitl_interface()
    
    if session_config:
        print("\n📊 Session Configuration:")
        print("="*80)
        for key, value in session_config.items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
