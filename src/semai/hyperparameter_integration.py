"""
Integration Guide: HyperparameterTesting Tool with Your Crew Agents

This shows how to integrate the HyperparameterTesting tool into your
meta_tuning_agent for automated hyperparameter optimization
"""

from crewai import Agent, Task, Crew, Process
from semai.tools import HyperparameterTestingTool
from crewai_tools import CodeInterpreterTool, FileReadTool, FileWriteTool
import numpy as np


# ============================================================================
# OPTION 1: Add tool to existing meta_tuning_agent
# ============================================================================

class MetaTuningAgentWithHP:
    """Updated meta_tuning_agent with hyperparameter testing"""
    
    @staticmethod
    def create_agent():
        """Create meta_tuning_agent with HyperparameterTesting tool"""
        
        return Agent(
            role="Senior ML Engineer 1 - Hyperparameter Tuning",
            goal=(
                "Conduct intelligent hyperparameter optimization using Random Search "
                "and Grid Search to find optimal parameters that maximize R² and "
                "minimize MSAE for derivative pricing models"
            ),
            backstory=(
                "You're a Senior ML Engineer with deep expertise in hyperparameter "
                "optimization and model selection. You systematically search the "
                "hyperparameter space using statistical approaches, evaluate models "
                "on validation data, and provide detailed analysis of parameter "
                "sensitivity. You understand the trade-offs between model complexity "
                "and performance."
            ),
            tools=[
                HyperparameterTestingTool(),  # NEW: Hyperparameter testing
                CodeInterpreterTool(),        # For analysis
                FileReadTool(),               # Load training data
                FileWriteTool(),              # Save results
            ],
            llm="meta-llama/Llama-3-70b-chat-hf",
            verbose=True
        )


# ============================================================================
# OPTION 2: Create dedicated hyperparameter tuning task
# ============================================================================

class HPTuningTask:
    """Task for hyperparameter tuning"""
    
    @staticmethod
    def create_task(agent):
        """Create hyperparameter tuning task"""
        
        return Task(
            description=(
                "Perform comprehensive hyperparameter tuning for derivative pricing models:\n"
                "1. Load the training and validation datasets\n"
                "2. Define parameter search spaces for each model type:\n"
                "   - Random Forest: n_estimators (50-250), max_depth (5-25), min_samples_split (2-10)\n"
                "   - Gradient Boosting: n_estimators (50-200), learning_rate (0.01-0.15), max_depth (3-10)\n"
                "   - Neural Networks: hidden_layers, learning_rate, batch_size, regularization\n"
                "3. Use Random Search for initial exploration (30 iterations per model)\n"
                "4. Use Grid Search for fine-tuning around best parameters\n"
                "5. Compare models based on R² and MSAE metrics\n"
                "6. Document the best configuration for each model\n"
                "7. Provide analysis of parameter sensitivity"
            ),
            expected_output=(
                "Comprehensive hyperparameter tuning report containing:\n"
                "- Best hyperparameters for each model type\n"
                "- Performance metrics (R², MSAE, RMSE, MAE) for best configurations\n"
                "- Comparison of model types\n"
                "- Parameter sensitivity analysis\n"
                "- Recommendations for deployment\n"
                "- Optimization history and visualization"
            ),
            agent=agent,
            human_input=False
        )


# ============================================================================
# OPTION 3: Multi-step hyperparameter optimization workflow
# ============================================================================

class HPOptimizationWorkflow:
    """Complete workflow for hyperparameter optimization"""
    
    @staticmethod
    def create_workflow():
        """Create full HP optimization workflow"""
        
        # Create agents
        hp_agent = MetaTuningAgentWithHP.create_agent()
        
        # Create tasks
        hp_task = HPTuningTask.create_task(hp_agent)
        
        # Create crew
        crew = Crew(
            agents=[hp_agent],
            tasks=[hp_task],
            process=Process.sequential,
            verbose=True
        )
        
        return crew
    
    @staticmethod
    def run_workflow(X_train, y_train, X_val, y_val):
        """Run the workflow"""
        crew = HPOptimizationWorkflow.create_workflow()
        result = crew.kickoff(
            inputs={
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
            }
        )
        return result


# ============================================================================
# OPTION 4: Direct tool usage in agent task
# ============================================================================

class DirectHPTuning:
    """Direct hyperparameter tuning without CrewAI orchestration"""
    
    @staticmethod
    def tune_random_forest(X_train, y_train, X_val, y_val):
        """Directly tune Random Forest using the tool"""
        from sklearn.ensemble import RandomForestRegressor
        from semai.tools import HyperparameterTestingTool
        
        tool = HyperparameterTestingTool()
        
        # Define parameter space
        param_config = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        
        # Run Random Search
        result = tool._run(
            strategy="random",
            model_class=RandomForestRegressor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_config=param_config,
            model_name="RandomForest",
            n_iter=30,
            scoring_metric="r_squared"
        )
        
        # Get best configuration
        best_params, best_result = tool.get_best_config("RandomForest")
        
        return best_params, best_result
    
    @staticmethod
    def tune_gradient_boosting(X_train, y_train, X_val, y_val):
        """Directly tune Gradient Boosting"""
        from sklearn.ensemble import GradientBoostingRegressor
        from semai.tools import HyperparameterTestingTool
        
        tool = HyperparameterTestingTool()
        
        param_config = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        }
        
        result = tool._run(
            strategy="random",
            model_class=GradientBoostingRegressor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_config=param_config,
            model_name="GradientBoosting",
            n_iter=30,
            scoring_metric="r_squared"
        )
        
        best_params, best_result = tool.get_best_config("GradientBoosting")
        
        return best_params, best_result
    
    @staticmethod
    def compare_all_models(X_train, y_train, X_val, y_val):
        """Compare all models"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from semai.tools import HyperparameterTestingTool
        
        tool = HyperparameterTestingTool()
        
        models = {
            'RandomForest': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [10, 15, 20],
                }
            },
            'GradientBoosting': {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 150, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                }
            }
        }
        
        results = {}
        for name, config in models.items():
            tool._run(
                strategy="random",
                model_class=config['class'],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                param_config=config['params'],
                model_name=name,
                n_iter=20,
                scoring_metric="r_squared"
            )
            
            best_params, best_result = tool.get_best_config(name)
            results[name] = {
                'params': best_params,
                'metrics': best_result.metrics
            }
        
        # Print comparison
        print("\nModel Comparison Results:")
        print("="*70)
        for name, data in results.items():
            print(f"\n{name}:")
            print(f"  Best Parameters: {data['params']}")
            print(f"  R²: {data['metrics']['r_squared']:.6f}")
            print(f"  MSAE: {data['metrics']['msae']:.6f}")
            print(f"  RMSE: {data['metrics']['rmse']:.6f}")
            print(f"  MAE: {data['metrics']['mae']:.6f}")
        
        # Get summary report
        print("\n" + tool.get_summary_report())
        
        # Plot results
        tool.plot_results(metric='r_squared', save_path='hp_comparison.png')
        tool.save_results('hp_comparison_results.json')
        
        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    print("Generating sample data...")
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # OPTION 1: Use direct tool
    print("\n" + "="*70)
    print("OPTION 1: Direct Hyperparameter Tuning")
    print("="*70)
    
    rf_params, rf_result = DirectHPTuning.tune_random_forest(X_train, y_train, X_val, y_val)
    print(f"Best RandomForest R²: {rf_result.metrics['r_squared']:.6f}")
    
    # OPTION 2: Compare all models
    print("\n" + "="*70)
    print("OPTION 2: Compare All Models")
    print("="*70)
    
    results = DirectHPTuning.compare_all_models(X_train, y_train, X_val, y_val)
    
    # OPTION 3: Use in CrewAI workflow
    # print("\nOPTION 3: CrewAI Workflow")
    # workflow = HPOptimizationWorkflow.create_workflow()
    # result = workflow.run_workflow(X_train, y_train, X_val, y_val)
