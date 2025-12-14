"""
Validation Crew - Model Risk Management and Compliance Validation

This crew validates the outputs of the Computational Crew through:
- Documentation compliance checking against institutional requirements
- Model replication to verify reproducibility
- Robustness testing under distribution shift
- Comprehensive compliance judgment
- Unified documentation synthesis
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool, FileWriterTool, DirectoryReadTool, PDFSearchTool
from typing import List


def _get_base_tools():
    """Get base tools available to all agents."""
    return [
        FileReadTool(),
        FileWriterTool(),
        DirectoryReadTool(),
    ]


def _get_cag_tool():
    """Lazy load CAG tool."""
    from semai.tools import CAGTool
    return CAGTool()


def _get_pdf_search_tool():
    """Get PDF search tool for compliance checking."""
    return PDFSearchTool()


@CrewBase
class ValidationCrew():
    """Validation Crew for Model Risk Management"""

    agents: List[BaseAgent]
    tasks: List[Task]
    _computational_doc_path: str = None
    
    # Override config paths for validation crew
    agents_config = 'config/validation_agents.yaml'
    tasks_config = 'config/validation_tasks.yaml'
    
    def set_computational_doc_path(self, path: str):
        """Set the path to Computational Crew documentation for validation."""
        self._computational_doc_path = path

    @agent
    def documentation_compliance_checker(self) -> Agent:
        config = self.agents_config['documentation_compliance_checker']
        return Agent(
            config=config,
            tools=_get_base_tools() + [_get_pdf_search_tool(), _get_cag_tool()],
            verbose=True
        )

    @agent
    def model_replication_agent(self) -> Agent:
        config = self.agents_config['model_replication_agent']
        return Agent(
            config=config,
            tools=_get_base_tools(),
            verbose=True
        )

    @agent
    def robustness_check_agent(self) -> Agent:
        config = self.agents_config['robustness_check_agent']
        return Agent(
            config=config,
            tools=_get_base_tools(),
            verbose=True
        )

    @agent
    def compliance_judge_agent(self) -> Agent:
        config = self.agents_config['compliance_judge_agent']
        return Agent(
            config=config,
            tools=_get_base_tools() + [_get_cag_tool()],
            verbose=True
        )

    @agent
    def validation_documentation_writer(self) -> Agent:
        config = self.agents_config['validation_documentation_writer']
        return Agent(
            config=config,
            tools=_get_base_tools(),
            verbose=True
        )

    @task
    def documentation_compliance_task(self) -> Task:
        return Task(
            config=self.tasks_config['documentation_compliance_task'],
        )

    @task
    def model_replication_task(self) -> Task:
        return Task(
            config=self.tasks_config['model_replication_task'],
        )

    @task
    def robustness_check_task(self) -> Task:
        return Task(
            config=self.tasks_config['robustness_check_task'],
        )

    @task
    def compliance_judgment_task(self) -> Task:
        return Task(
            config=self.tasks_config['compliance_judgment_task'],
        )

    @task
    def comprehensive_documentation_task(self) -> Task:
        return Task(
            config=self.tasks_config['comprehensive_documentation_task'],
            output_file='ComprehensiveSummary.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Validation crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
