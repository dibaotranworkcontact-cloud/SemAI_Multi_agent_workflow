"""
Interactive crew executor with per-task feedback and control.
Allows users to pause between tasks, provide feedback, or end execution.
"""

from typing import Any, Dict, List
from crewai import Crew, Process, Task
from semai.task_feedback_handler import get_task_feedback_handler


class InteractiveCrewExecutor:
    """
    Wraps crew execution to provide per-task user control.
    Implements custom sequential execution with feedback prompts between tasks.
    """
    
    def __init__(self, crew, inputs: Dict[str, Any], feedback_handler=None):
        """
        Initialize the interactive executor.
        
        Args:
            crew: CrewAI Crew instance
            inputs: Input dictionary for crew execution
            feedback_handler: Optional feedback handler instance (kept separate from inputs)
        """
        self.crew = crew
        self.inputs = inputs
        self.feedback_handler = feedback_handler or get_task_feedback_handler()
        self.enable_feedback = inputs.get('enable_task_feedback', False)
        self.task_results = {}
        self.execution_halted = False
        self.task_count = 0
        self.completed_tasks = 0
        self.context = {}
    
    def _execute_task_with_context(self, task: Task, previous_tasks: list = None):
        """
        Execute a single task with previously completed tasks for context.
        
        Args:
            task: Task to execute
            previous_tasks: List of previous tasks to include in crew context
            
        Returns:
            Task output/result
        """
        if previous_tasks is None:
            previous_tasks = []
        
        # Execute task with available context
        try:
            # Create a crew with this task and all previous tasks for context
            tasks_to_execute = previous_tasks + [task]
            
            temp_crew = Crew(
                agents=[t.agent for t in tasks_to_execute],
                tasks=tasks_to_execute,
                process=Process.sequential,
                verbose=True
            )
            
            result = temp_crew.kickoff(inputs=self.inputs)
            # Return only the current task result
            return result
        
        except Exception as e:
            raise Exception(f"Error executing task {task.description}: {e}")
    
    def _execute_sequential_with_feedback(self):
        """
        Execute tasks sequentially with user feedback between each task.
        Returns the final result or halts early if user requests.
        """
        results = {}
        completed_tasks = []
        
        for idx, task in enumerate(self.crew.tasks):
            self.completed_tasks = idx + 1
            task_num = self.completed_tasks
            total = self.task_count
            
            print("\n" + "="*80)
            print(f"⏳ EXECUTING TASK {task_num}/{total}")
            print("="*80)
            print(f"Task: {task.description[:60]}...")
            print("="*80 + "\n")
            
            try:
                # Execute task with all previous tasks for context
                task_result = self._execute_task_with_context(task, completed_tasks)
                results[task.description] = task_result
                completed_tasks.append(task)
                
                # Convert result to string for display
                task_output_str = str(task_result)
                
                print("\n" + "="*80)
                print(f"✅ TASK {task_num}/{total} COMPLETED")
                print("="*80 + "\n")
                
                # Prompt for feedback
                feedback_result = self.feedback_handler.prompt_for_feedback(
                    task.description, 
                    task_output_str
                )
                
                # Check if user wants to end execution
                if feedback_result['action'] == 'end':
                    self.execution_halted = True
                    print(f"\n⏹️  Execution stopped by user after {task_num}/{total} tasks.\n")
                    break
                
                # Store feedback if provided
                if feedback_result['feedback_text']:
                    self.feedback_handler.store_task_feedback(
                        task.description, 
                        feedback_result['feedback_text']
                    )
            
            except Exception as e:
                print(f"\n❌ Error executing task {task_num}: {e}\n")
                raise
        
        return results
    
    def execute(self):
        """
        Execute crew tasks with interactive feedback between each task.
        Returns the final crew result.
        """
        if not self.enable_feedback:
            # If feedback not enabled, just run normally
            return self.crew.kickoff(inputs=self.inputs)
        
        # Setup for interactive execution
        self.task_count = len(self.crew.tasks)
        
        # Execute with feedback capability
        print("\n📋 Executing crew with per-task feedback enabled...\n")
        
        try:
            print("\n" + "="*80)
            print("⏳ EXECUTING ML PIPELINE WITH TASK FEEDBACK")
            print("="*80)
            print(f"Total tasks to execute: {self.task_count}")
            print("You can provide feedback or stop execution after each task.")
            print("="*80 + "\n")
            
            # Execute tasks sequentially with feedback
            results = self._execute_sequential_with_feedback()
            
            if self.execution_halted:
                print(f"\n⏹️  Execution halted after {self.completed_tasks}/{self.task_count} tasks.\n")
            else:
                print(f"\n✅ All {self.completed_tasks} tasks completed successfully!\n")
            
            return self._compile_results(results)
        
        except Exception as e:
            print(f"\n❌ Error during crew execution: {e}\n")
            raise
    
    def _compile_results(self, task_results):
        """Compile results from task executions."""
        result = {
            'tasks_completed': self.completed_tasks,
            'total_tasks': self.task_count,
            'execution_halted': self.execution_halted,
            'crew_output': task_results,
            'feedbacks': self.feedback_handler.get_all_feedbacks()
        }
        return result


def execute_crew_with_feedback(crew, inputs: Dict[str, Any], feedback_handler=None):
    """
    Execute a crew with interactive task-level feedback.
    
    Args:
        crew: CrewAI Crew instance
        inputs: Input dictionary for crew execution
        feedback_handler: Optional feedback handler instance (not passed to crew inputs)
        
    Returns:
        Execution result
    """
    executor = InteractiveCrewExecutor(crew, inputs, feedback_handler)
    return executor.execute()
