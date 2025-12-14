"""
Example Usage of HyperparameterTesting Tool
Demonstrates Random Search, Grid Search, and Result Analysis
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from semai.tools.hyperparameter_testing import (
    HyperparameterTestingTool,
    quick_hyperparameter_test,
    PerformanceMetrics
)


# ============================================================================
# EXAMPLE 1: Quick Hyperparameter Test with Random Forest
# ============================================================================

def example_1_quick_test():
    """Simple quick test for hyperparameter tuning"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Hyperparameter Test")
    print("="*70 + "\n")
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter ranges
    param_ranges = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    # Run quick test
    best_params, best_r2 = quick_hyperparameter_test(
        model_class=RandomForestRegressor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_ranges=param_ranges,
        n_iter=20,
        model_name="RandomForest"
    )
    
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best R² Score: {best_r2:.6f}")


# ============================================================================
# EXAMPLE 2: Random Search with Multiple Models
# ============================================================================

def example_2_random_search():
    """Test multiple models with random search"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Random Search with Multiple Models")
    print("="*70 + "\n")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize tool
    tool = HyperparameterTestingTool()
    
    # Test Random Forest
    print("Testing Random Forest...")
    rf_params = {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    tool._run(
        strategy="random",
        model_class=RandomForestRegressor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_config=rf_params,
        model_name="RandomForest",
        n_iter=30,
        scoring_metric="r_squared"
    )
    
    # Test Gradient Boosting
    print("\nTesting Gradient Boosting...")
    gb_params = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
    }
    
    tool._run(
        strategy="random",
        model_class=GradientBoostingRegressor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_config=gb_params,
        model_name="GradientBoosting",
        n_iter=30,
        scoring_metric="r_squared"
    )
    
    # Print summary
    print(tool.get_summary_report())
    
    # Save results
    tool.save_results("hyperparameter_results.json")
    
    # Plot results
    tool.plot_results(metric="r_squared", save_path="hyperparameter_results.png")


# ============================================================================
# EXAMPLE 3: Grid Search for Fine-tuning
# ============================================================================

def example_3_grid_search():
    """Use grid search for thorough fine-tuning"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Grid Search for Fine-tuning")
    print("="*70 + "\n")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize tool
    tool = HyperparameterTestingTool()
    
    # Define parameter grid (smaller grid for grid search)
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
    }
    
    result = tool._run(
        strategy="grid",
        model_class=RandomForestRegressor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_config=param_grid,
        model_name="RandomForest_GridSearch",
        scoring_metric="r_squared"
    )
    
    print(result)


# ============================================================================
# EXAMPLE 4: Custom Metric Optimization (MSAE)
# ============================================================================

def example_4_custom_metrics():
    """Optimize for different metrics (MSAE, R², RMSE)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Optimizing for Custom Metrics")
    print("="*70 + "\n")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tool = HyperparameterTestingTool()
    
    param_ranges = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
    }
    
    # Optimize for MSAE (lower is better)
    print("Optimizing for MSAE...")
    tool._run(
        strategy="random",
        model_class=RandomForestRegressor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_config=param_ranges,
        model_name="Model_MSAE",
        n_iter=20,
        scoring_metric="msae"  # Optimize for MSAE
    )
    
    # Get results
    df = tool.get_results_dataframe()
    print(f"\nMSAE - Min: {df['msae'].min():.6f}, Max: {df['msae'].max():.6f}")
    print(f"R² - Min: {df['r_squared'].min():.6f}, Max: {df['r_squared'].max():.6f}")


# ============================================================================
# EXAMPLE 5: Analysis and Comparison
# ============================================================================

def example_5_comparison():
    """Compare different hyperparameter configurations"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Comparing Hyperparameter Configurations")
    print("="*70 + "\n")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tool = HyperparameterTestingTool()
    
    # Test with different parameter ranges
    configs = [
        {
            'name': 'Small Model',
            'params': {
                'n_estimators': [50],
                'max_depth': [5],
            }
        },
        {
            'name': 'Medium Model',
            'params': {
                'n_estimators': [100, 150],
                'max_depth': [10, 15],
            }
        },
        {
            'name': 'Large Model',
            'params': {
                'n_estimators': [200, 250],
                'max_depth': [20, 25],
            }
        }
    ]
    
    for config in configs:
        print(f"Testing {config['name']}...")
        tool._run(
            strategy="random",
            model_class=RandomForestRegressor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_config=config['params'],
            model_name=config['name'],
            n_iter=15,
            scoring_metric="r_squared"
        )
    
    # Compare results
    df = tool.get_results_dataframe()
    comparison = df.groupby('model_name')[['r_squared', 'msae', 'rmse', 'mae']].agg(['min', 'max', 'mean'])
    
    print("\nComparison of Configurations:")
    print(comparison)


# ============================================================================
# EXAMPLE 6: Integration with CrewAI Agent
# ============================================================================

def example_6_crewai_integration():
    """Show how to use HyperparameterTesting tool in a CrewAI agent"""
    print("\n" + "="*70)
    print("EXAMPLE 6: CrewAI Integration")
    print("="*70 + "\n")
    
    from crewai import Agent
    
    # Create hyperparameter tuning agent
    hyperparameter_tuning_agent = Agent(
        role="Hyperparameter Tuning Specialist",
        goal="Find optimal hyperparameters for models using random and grid search",
        backstory="Expert in hyperparameter optimization with deep knowledge of model performance metrics",
        tools=[HyperparameterTestingTool()],
        llm="gpt-4-turbo",
        verbose=True
    )
    
    print("Agent Created Successfully!")
    print(f"Agent Role: {hyperparameter_tuning_agent.role}")
    print(f"Tools: {[tool.name for tool in hyperparameter_tuning_agent.tools]}")
    
    return hyperparameter_tuning_agent


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run examples
    example_1_quick_test()
    example_2_random_search()
    example_3_grid_search()
    example_4_custom_metrics()
    example_5_comparison()
    # example_6_crewai_integration()  # Uncomment to test CrewAI integration
