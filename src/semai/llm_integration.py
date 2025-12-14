"""
Crew Integration with LLM Models
Integrates Meta Llama 3 70B and Deepseek V3.1 with CrewAI
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from .llm_config import MODELS, client

# Set the API key in environment
API_KEY = "246c6b4fa5304217d38fedd55678e546b5d5971acd7f470d7af1911b0aa102d5"
os.environ["TOGETHER_API_KEY"] = API_KEY

class ModelSelector:
    """Helper class to select which model to use for agents"""
    
    def __init__(self):
        self.llama_model = MODELS["llama_3_70b"]["name"]
        self.deepseek_model = MODELS["deepseek_v3_1"]["name"]
    
    def get_analytical_model(self):
        """Use Llama for analysis tasks"""
        return self.llama_model
    
    def get_reasoning_model(self):
        """Use Deepseek for reasoning tasks"""
        return self.deepseek_model
    
    def get_model(self, model_type: str = "analytical"):
        """Get model based on type"""
        if model_type == "reasoning":
            return self.get_reasoning_model()
        return self.get_analytical_model()

# Initialize model selector
model_selector = ModelSelector()

# Custom agent configuration wrapper
def create_agent_config(role: str, goal: str, backstory: str, llm_model: str = None):
    """Create agent config with LLM model"""
    if llm_model is None:
        llm_model = model_selector.get_analytical_model()
    
    return {
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "llm": llm_model
    }

if __name__ == "__main__":
    print("LLM Models Integrated with Crew Successfully!")
    print(f"\nAvailable Models:")
    print(f"- Llama 3 70B: {model_selector.get_analytical_model()}")
    print(f"- Deepseek V3.1: {model_selector.get_reasoning_model()}")
