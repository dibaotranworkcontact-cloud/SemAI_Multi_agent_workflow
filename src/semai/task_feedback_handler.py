"""
Task Feedback Handler for inter-task user interaction.
Allows users to provide feedback, continue, or end execution after each task.
"""


class TaskFeedbackHandler:
    """
    Handles user feedback and control flow after each task execution.
    """
    
    def __init__(self):
        self.task_feedbacks = {}
        self.execution_paused = False
        self.should_end = False
    
    def prompt_for_feedback(self, task_name: str, task_output: str):
        """
        Prompt user for feedback after task execution.
        
        Args:
            task_name: Name of the completed task
            task_output: Output/result of the task
            
        Returns:
            dict with keys: 'action' ('continue', 'end', or 'feedback'), 'feedback_text' (if action is 'feedback')
        """
        print(f"📊 Task Output (first 300 chars):\n{task_output[:300]}...")
        print("\n" + "="*80)
        print("WHAT WOULD YOU LIKE TO DO?")
        print("="*80)
        print("  [C] Continue to next task")
        print("  [E] End execution now")
        print("  [F] Provide feedback for this task")
        print("="*80)
        
        while True:
            choice = input("\nEnter your choice (C/E/F): ").strip().lower()
            
            if choice == 'c':
                print("✅ Continuing to next task...\n")
                return {'action': 'continue', 'feedback_text': ''}
            
            elif choice == 'e':
                print("⏹️  Execution stopped by user.\n")
                self.should_end = True
                return {'action': 'end', 'feedback_text': ''}
            
            elif choice == 'f':
                feedback = input("\n📝 Enter your feedback: ").strip()
                if feedback:
                    self.task_feedbacks[task_name] = feedback
                    print("✅ Feedback saved. Continuing to next task...\n")
                else:
                    print("⚠️  No feedback entered. Continuing anyway...\n")
                return {'action': 'feedback', 'feedback_text': feedback}
            
            else:
                print("⚠️  Invalid choice. Please enter C, E, or F.")
    
    def store_task_feedback(self, task_name: str, feedback: str):
        """Store feedback for a specific task."""
        self.task_feedbacks[task_name] = feedback
    
    def get_task_feedback(self, task_name: str):
        """Retrieve feedback for a specific task."""
        return self.task_feedbacks.get(task_name, '')
    
    def get_all_feedbacks(self):
        """Get all task feedbacks."""
        return self.task_feedbacks
    
    def should_continue_execution(self):
        """Check if execution should continue."""
        return not self.should_end
    
    def reset(self):
        """Reset feedback handler for new execution."""
        self.task_feedbacks = {}
        self.execution_paused = False
        self.should_end = False


# Global instance
_feedback_handler = None


def get_task_feedback_handler():
    """Get or create global task feedback handler."""
    global _feedback_handler
    if _feedback_handler is None:
        _feedback_handler = TaskFeedbackHandler()
    return _feedback_handler


def reset_feedback_handler():
    """Reset the global feedback handler."""
    global _feedback_handler
    if _feedback_handler is not None:
        _feedback_handler.reset()
