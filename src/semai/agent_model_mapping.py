"""
Agent to LLM Model Mapping Configuration
Maps each agent to its designated LLM model as per the brief
"""

from llm_config import MODELS

# Agent-to-Model mapping based on the brief
AGENT_MODEL_MAPPING = {
    # GPT-3.5 Turbo agents
    "data_extraction_agent": {
        "model": "gpt-4-mini",
        "description": "Data Extraction Agent",
        "provider": "OpenAI"
    },
    
    # Llama3 agent
    "eda_agent": {
        "model": MODELS["meta-llama/Llama-3-70b-chat-hf"]["name"],
        "description": "EDA Agent powered by Llama3",
        "provider": "Together AI"
    },
    
    # GPT-3.5 Turbo agents
    "feature_engineering_agent": {
        "model": "gpt-4-mini",
        "description": "Feature Engineering Agent",
        "provider": "OpenAI"
    },
    
    # Senior ML Engineer - GPT-3.5 Turbo
    "meta_tuning_agent": {
        "model": "gpt-4-mini",
        "description": "Senior ML Engineer for Meta-Tuning",
        "provider": "OpenAI"
    },
    
    # Senior ML Engineer - GPT-3.5 Turbo
    "model_training_agent": {
        "model": "gpt-4-mini",
        "description": "Senior ML Engineer for Model Training",
        "provider": "OpenAI"
    },
    
    # Model Evaluation Agent
    "model_evaluation_agent": {
        "model": MODELS["meta-llama/Llama-3-70b-chat-hf"]["name"],
        "description": "Model Evaluation Agent",
        "provider": "Together AI"
    },
    
    # DeepSeek-R1 for Judge/Manager
    "judge_agent": {
        "model": "deepseek-ai/DeepSeek-R1",
        "description": "Judge Agent - Manager for Quality Assessment",
        "provider": "Together AI"
    },
    
    # DeepSeek-R1 for Documentation Writer
    "documentation_writer": {
        "model": "deepseek-ai/DeepSeek-R1",
        "description": "Documentation Writer Agent",
        "provider": "Together AI"
    }
}

def get_agent_model(agent_name: str) -> str:
    """Get the designated model for an agent"""
    if agent_name in AGENT_MODEL_MAPPING:
        return AGENT_MODEL_MAPPING[agent_name]["model"]
    return MODELS["llama_3_70b"]["name"]  # Default fallback

def get_agent_config(agent_name: str) -> dict:
    """Get full configuration for an agent"""
    return AGENT_MODEL_MAPPING.get(agent_name, {})

def print_agent_mapping():
    """Print the agent-to-model mapping"""
    print("\n" + "="*70)
    print("AGENT TO LLM MODEL MAPPING")
    print("="*70 + "\n")
    
    for agent, config in AGENT_MODEL_MAPPING.items():
        print(f"Agent: {config['description']}")
        print(f"  - Name: {agent}")
        print(f"  - Model: {config['model']}")
        print(f"  - Provider: {config['provider']}")
        print()

if __name__ == "__main__":
    print_agent_mapping()
