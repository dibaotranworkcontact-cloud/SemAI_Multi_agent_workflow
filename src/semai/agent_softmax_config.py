"""
Agent Softmax Integration Module
Applies softmax metrics to agent LLM initialization for consistent temperature scaling.
"""

from typing import Dict, Optional
from semai.softmax_metrics import get_global_softmax_metrics, SoftmaxMetrics

class AgentSoftmaxConfig:
    """
    Configuration manager for applying softmax metrics to CrewAI agents.
    Ensures all agents use consistent temperature scaling (0.2) for deterministic output.
    """
    
    # Global softmax temperature for all agents
    AGENT_SOFTMAX_TEMPERATURE = 0.2
    
    def __init__(self):
        """Initialize agent softmax configuration."""
        self.softmax_metrics = get_global_softmax_metrics(temperature=self.AGENT_SOFTMAX_TEMPERATURE)
        self.agent_configs = {}
    
    def get_agent_llm_config(self, agent_name: str, base_config: Dict) -> Dict:
        """
        Get LLM configuration for an agent with softmax metrics applied.
        
        Args:
            agent_name: Name of the agent
            base_config: Base LLM configuration from agents.yaml
        
        Returns:
            Updated configuration with softmax metrics
        """
        config = base_config.copy()
        
        # Apply softmax temperature (0.2 = prioritize high-scoring tokens, reduce randomness)
        config['temperature'] = self.AGENT_SOFTMAX_TEMPERATURE
        config['top_k'] = 50  # Keep top 50 tokens
        config['top_p'] = 0.9  # Nucleus sampling at 90%
        
        # Add softmax-specific metadata
        config['softmax_enabled'] = True
        config['softmax_temperature'] = self.AGENT_SOFTMAX_TEMPERATURE
        config['sampling_method'] = 'nucleus'  # Use nucleus sampling for balanced output
        
        # Store configuration
        self.agent_configs[agent_name] = config
        
        return config
    
    def get_all_agents_config(self, agents_dict: Dict) -> Dict:
        """
        Apply softmax metrics to all agents.
        
        Args:
            agents_dict: Dictionary of all agent configurations
        
        Returns:
            Updated agent configurations with softmax applied
        """
        updated_configs = {}
        
        for agent_name, agent_config in agents_dict.items():
            updated_configs[agent_name] = self.get_agent_llm_config(agent_name, agent_config)
        
        return updated_configs
    
    def get_softmax_metrics(self) -> SoftmaxMetrics:
        """Get the global softmax metrics instance."""
        return self.softmax_metrics
    
    def get_config_summary(self) -> Dict:
        """
        Get summary of softmax configuration applied to all agents.
        
        Returns:
            Configuration summary
        """
        return {
            "softmax_temperature": self.AGENT_SOFTMAX_TEMPERATURE,
            "total_agents_configured": len(self.agent_configs),
            "sampling_method": "nucleus",
            "top_k": 50,
            "top_p": 0.9,
            "agents": list(self.agent_configs.keys()),
            "metrics_summary": self.softmax_metrics.get_metrics_summary()
        }


# Global instance
_agent_softmax_config = None

def get_agent_softmax_config() -> AgentSoftmaxConfig:
    """
    Get or create global agent softmax configuration.
    
    Returns:
        Global AgentSoftmaxConfig instance
    """
    global _agent_softmax_config
    if _agent_softmax_config is None:
        _agent_softmax_config = AgentSoftmaxConfig()
    return _agent_softmax_config

def apply_softmax_to_all_agents(agents_dict: Dict) -> Dict:
    """
    Apply softmax metrics to all agents in the dictionary.
    
    Args:
        agents_dict: Dictionary of agent configurations
    
    Returns:
        Updated agent configurations
    """
    config = get_agent_softmax_config()
    return config.get_all_agents_config(agents_dict)

def print_softmax_config():
    """Print current softmax configuration for all agents."""
    config = get_agent_softmax_config()
    summary = config.get_config_summary()
    
    print("\n" + "="*70)
    print("🌡️  AGENT SOFTMAX CONFIGURATION")
    print("="*70)
    print(f"Temperature: {summary['softmax_temperature']} (prioritizes high-scoring tokens)")
    print(f"Sampling Method: {summary['sampling_method']}")
    print(f"Top-K: {summary['top_k']}")
    print(f"Top-P (Nucleus): {summary['top_p']}")
    print(f"Agents Configured: {summary['total_agents_configured']}")
    
    if summary['agents']:
        print(f"\nAgents with Softmax Applied:")
        for agent in summary['agents']:
            print(f"  ✓ {agent}")
    
    metrics = summary['metrics_summary']
    if metrics:
        print(f"\nMetrics:")
        print(f"  Avg Confidence: {metrics.get('avg_confidence', 0):.4f}")
        print(f"  Avg Entropy: {metrics.get('avg_entropy', 0):.4f}")
        print(f"  Total Predictions: {metrics.get('total_predictions', 0)}")
    
    print("="*70 + "\n")
