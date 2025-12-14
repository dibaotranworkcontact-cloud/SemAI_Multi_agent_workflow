from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool, FileWriterTool, DirectoryReadTool, CodeInterpreterTool
from typing import List
from semai.agent_softmax_config import get_agent_softmax_config

# Initialize softmax configuration for all agents
_agent_softmax_config = get_agent_softmax_config()

# Import tools lazily to avoid circular imports
def _get_load_csv_tool():
    """Lazy load the tool to avoid circular imports."""
    from semai.tools.custom_tool import LoadCSVTool
    return LoadCSVTool()

def _get_hyperparameter_tool():
    """Lazy load hyperparameter testing tool."""
    from semai.tools import HyperparameterTestingTool
    return HyperparameterTestingTool()

def _get_cag_tool():
    """Lazy load CAG tool."""
    from semai.tools import CAGTool
    return CAGTool()

def _get_base_tools():
    """Get base tools available to all agents."""
    return [
        FileReadTool(),
        FileWriterTool(),
        DirectoryReadTool(),
    ]

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Semai():
    """Semai Modeling Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    _selected_model: str = None  # Track selected model

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    
    def update_agent_models(self, model_name: str):
        """
        Update all agents to use the specified model.
        
        Args:
            model_name: The LLM model to use for all agents
        """
        self._selected_model = model_name
        # Update config for all agents
        for agent_name in self.agents_config:
            if 'llm' in self.agents_config[agent_name]:
                self.agents_config[agent_name]['llm'] = model_name
    
    @agent
    def data_extraction_agent(self) -> Agent:
        config = self.agents_config['data_extraction_agent']
        # Apply selected model if available
        if self._selected_model:
            config['llm'] = self._selected_model
        # Apply softmax metrics configuration
        softmax_config = _agent_softmax_config.get_agent_llm_config('data_extraction_agent', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools() + [_get_load_csv_tool()],
            verbose=True
        )

    @agent
    def eda_agent(self) -> Agent:
        config = self.agents_config['eda_agent']
        if self._selected_model:
            config['llm'] = self._selected_model
        softmax_config = _agent_softmax_config.get_agent_llm_config('eda_agent', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools(),
            verbose=True
        )

    @agent
    def feature_engineering_agent(self) -> Agent:
        config = self.agents_config['feature_engineering_agent']
        if self._selected_model:
            config['llm'] = self._selected_model
        softmax_config = _agent_softmax_config.get_agent_llm_config('feature_engineering_agent', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools(),
            verbose=True
        )

    @agent
    def meta_tuning_agent(self) -> Agent:
        config = self.agents_config['meta_tuning_agent']
        if self._selected_model:
            config['llm'] = self._selected_model
        softmax_config = _agent_softmax_config.get_agent_llm_config('meta_tuning_agent', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools() + [_get_hyperparameter_tool()],
            verbose=True
        )

    @agent
    def model_training_agent(self) -> Agent:
        config = self.agents_config['model_training_agent']
        if self._selected_model:
            config['llm'] = self._selected_model
        softmax_config = _agent_softmax_config.get_agent_llm_config('model_training_agent', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools(),
            verbose=True
        )

    @agent
    def model_evaluation_agent(self) -> Agent:
        config = self.agents_config['model_evaluation_agent']
        if self._selected_model:
            config['llm'] = self._selected_model
        softmax_config = _agent_softmax_config.get_agent_llm_config('model_evaluation_agent', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools(),
            verbose=True
        )

    @agent
    def documentation_writer(self) -> Agent:
        config = self.agents_config['documentation_writer']
        if self._selected_model:
            config['llm'] = self._selected_model
        softmax_config = _agent_softmax_config.get_agent_llm_config('documentation_writer', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools(),
            verbose=True
        )

    @agent
    def judge_agent(self) -> Agent:
        config = self.agents_config['judge_agent']
        if self._selected_model:
            config['llm'] = self._selected_model
        softmax_config = _agent_softmax_config.get_agent_llm_config('judge_agent', config)
        return Agent(
            config=softmax_config,
            tools=_get_base_tools() + [_get_cag_tool()],
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    
    @task
    def data_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_extraction_task'], # type: ignore[index]
        )

    @task
    def eda_task(self) -> Task:
        return Task(
            config=self.tasks_config['eda_task'], # type: ignore[index]
        )

    @task
    def feature_engineering_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_engineering_task'], # type: ignore[index]
        )

    @task
    def meta_tuning_task(self) -> Task:
        return Task(
            config=self.tasks_config['meta_tuning_task'], # type: ignore[index]
        )

    @task
    def model_training_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_training_task'], # type: ignore[index]
        )

    @task
    def model_evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_evaluation_task'], # type: ignore[index]
        )

    @task
    def judge_task(self) -> Task:
        return Task(
            config=self.tasks_config['judge_task'], # type: ignore[index]
        )

    @task
    def documentation_task(self) -> Task:
        return Task(
            config=self.tasks_config['documentation_task'], # type: ignore[index]
            output_file='modeling_documentation.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Semai modeling crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
