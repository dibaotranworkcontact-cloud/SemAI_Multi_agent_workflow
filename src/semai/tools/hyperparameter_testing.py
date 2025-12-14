"""
Hyperparameter Testing Tool for Model Optimization
Provides RandomSearch, GridSearch, and Bayesian optimization for hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import BaseTool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HyperparameterResult:
    """Store hyperparameter test results"""
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str
    model_name: str
    strategy: str
    
    def __str__(self):
        return f"HP: {self.hyperparameters}\nMetrics: {self.metrics}"


class PerformanceMetrics:
    """Calculate and manage performance metrics"""
    
    @staticmethod
    def calculate_msae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Absolute Error (MSAE)
        MSAE = mean(|y_true - y_pred|^2)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSAE score (lower is better)
        """
        return np.mean(np.square(np.abs(y_true - y_pred)))
    
    @staticmethod
    def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² (Coefficient of Determination)
        R² = 1 - (SS_res / SS_tot)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² score (higher is better, max 1.0)
        """
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE score (lower is better)
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE score (lower is better)
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def get_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Get all available metrics"""
        return {
            'msae': PerformanceMetrics.calculate_msae(y_true, y_pred),
            'r_squared': PerformanceMetrics.calculate_r_squared(y_true, y_pred),
            'rmse': PerformanceMetrics.calculate_rmse(y_true, y_pred),
            'mae': PerformanceMetrics.calculate_mae(y_true, y_pred),
        }


class RandomSearchOptimizer:
    """Random search for hyperparameter optimization"""
    
    def __init__(self, param_distributions: Dict[str, List], n_iter: int = 10, 
                 random_state: int = 42):
        """
        Initialize RandomSearch optimizer
        
        Args:
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of iterations
            random_state: Random seed
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.results: List[HyperparameterResult] = []
    
    def search(self, 
               model_class: type,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               model_name: str = "Model",
               scoring_metric: str = "r_squared") -> Tuple[Dict, HyperparameterResult]:
        """
        Perform random search
        
        Args:
            model_class: Model class to instantiate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the model
            scoring_metric: Metric to optimize ('r_squared' or 'msae')
            
        Returns:
            Tuple of (best_hyperparameters, best_result)
        """
        logger.info(f"Starting Random Search with {self.n_iter} iterations")
        
        param_sampler = ParameterSampler(
            self.param_distributions,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        best_score = -np.inf if scoring_metric == 'r_squared' else np.inf
        best_hyperparams = None
        best_result = None
        
        for iteration, params in enumerate(param_sampler, 1):
            try:
                # Train model
                model = model_class(**params)
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                metrics = PerformanceMetrics.get_all_metrics(y_val, y_pred)
                
                # Store result
                result = HyperparameterResult(
                    hyperparameters=params,
                    metrics=metrics,
                    timestamp=datetime.now().isoformat(),
                    model_name=model_name,
                    strategy="random_search"
                )
                self.results.append(result)
                
                # Check if best
                current_score = metrics[scoring_metric]
                is_better = (current_score > best_score) if scoring_metric == 'r_squared' else (current_score < best_score)
                
                if is_better:
                    best_score = current_score
                    best_hyperparams = params
                    best_result = result
                
                logger.info(f"Iteration {iteration}/{self.n_iter} - {scoring_metric}: {current_score:.6f}")
            
            except Exception as e:
                logger.warning(f"Iteration {iteration} failed: {str(e)}")
                continue
        
        logger.info(f"Random Search Complete. Best {scoring_metric}: {best_score:.6f}")
        return best_hyperparams, best_result


class GridSearchOptimizer:
    """Grid search for hyperparameter optimization"""
    
    def __init__(self, param_grid: Dict[str, List]):
        """
        Initialize GridSearch optimizer
        
        Args:
            param_grid: Dictionary of parameter grid
        """
        self.param_grid = param_grid
        self.results: List[HyperparameterResult] = []
    
    def search(self,
               model_class: type,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               model_name: str = "Model",
               scoring_metric: str = "r_squared") -> Tuple[Dict, HyperparameterResult]:
        """
        Perform grid search
        
        Args:
            model_class: Model class to instantiate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the model
            scoring_metric: Metric to optimize
            
        Returns:
            Tuple of (best_hyperparameters, best_result)
        """
        param_grid_instance = ParameterGrid(self.param_grid)
        total_combinations = len(list(param_grid_instance))
        
        logger.info(f"Starting Grid Search with {total_combinations} combinations")
        
        best_score = -np.inf if scoring_metric == 'r_squared' else np.inf
        best_hyperparams = None
        best_result = None
        
        for iteration, params in enumerate(ParameterGrid(self.param_grid), 1):
            try:
                # Train model
                model = model_class(**params)
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                metrics = PerformanceMetrics.get_all_metrics(y_val, y_pred)
                
                # Store result
                result = HyperparameterResult(
                    hyperparameters=params,
                    metrics=metrics,
                    timestamp=datetime.now().isoformat(),
                    model_name=model_name,
                    strategy="grid_search"
                )
                self.results.append(result)
                
                # Check if best
                current_score = metrics[scoring_metric]
                is_better = (current_score > best_score) if scoring_metric == 'r_squared' else (current_score < best_score)
                
                if is_better:
                    best_score = current_score
                    best_hyperparams = params
                    best_result = result
                
                logger.info(f"Combination {iteration}/{total_combinations} - {scoring_metric}: {current_score:.6f}")
            
            except Exception as e:
                logger.warning(f"Combination {iteration} failed: {str(e)}")
                continue
        
        logger.info(f"Grid Search Complete. Best {scoring_metric}: {best_score:.6f}")
        return best_hyperparams, best_result


class HyperparameterTestingTool(BaseTool):
    """
    CrewAI Tool for hyperparameter testing and optimization
    Supports Random Search, Grid Search, and comprehensive metrics
    """
    
    name: str = "Hyperparameter Testing Tool"
    description: str = (
        "Test hyperparameters with Random Search or Grid Search. "
        "Evaluates model performance using MSAE, R², RMSE, and MAE metrics. "
        "Returns optimal hyperparameters and performance metrics."
    )
    
    # Use class-level attributes that Pydantic can handle
    results_history: List = []
    best_configs: Dict = {}
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, 
             strategy: str = "random",
             model_class = None,
             X_train = None,
             y_train = None,
             X_val = None,
             y_val = None,
             param_config: Dict = None,
             model_name: str = "Model",
             n_iter: int = 20,
             scoring_metric: str = "r_squared") -> str:
        """
        Run hyperparameter optimization
        
        Args:
            strategy: 'random' or 'grid'
            model_class: Model class to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_config: Parameter configuration
            model_name: Name of model being optimized
            n_iter: Number of iterations (for random search)
            scoring_metric: Metric to optimize ('r_squared' or 'msae')
            
        Returns:
            Result summary string
        """
        try:
            if strategy.lower() == "random":
                optimizer = RandomSearchOptimizer(param_config, n_iter=n_iter)
                best_params, best_result = optimizer.search(
                    model_class, X_train, y_train, X_val, y_val,
                    model_name, scoring_metric
                )
                self.results_history.extend(optimizer.results)
            
            elif strategy.lower() == "grid":
                optimizer = GridSearchOptimizer(param_config)
                best_params, best_result = optimizer.search(
                    model_class, X_train, y_train, X_val, y_val,
                    model_name, scoring_metric
                )
                self.results_history.extend(optimizer.results)
            
            else:
                return f"Error: Unknown strategy '{strategy}'. Use 'random' or 'grid'."
            
            # Store best config
            self.best_configs[model_name] = (best_params, best_result)
            
            # Format result
            result_str = self._format_results(best_params, best_result, strategy)
            return result_str
        
        except Exception as e:
            logger.error(f"Error during hyperparameter testing: {str(e)}")
            return f"Error: {str(e)}"
    
    def _format_results(self, 
                       best_params: Dict,
                       best_result: HyperparameterResult,
                       strategy: str) -> str:
        """Format results for display"""
        metrics_str = "\n  ".join([f"{k}: {v:.6f}" for k, v in best_result.metrics.items()])
        params_str = "\n  ".join([f"{k}: {v}" for k, v in best_params.items()])
        
        return f"""
Hyperparameter Optimization Results
{'='*60}
Strategy: {strategy.upper()}
Model: {best_result.model_name}

Best Hyperparameters:
  {params_str}

Performance Metrics:
  {metrics_str}

Timestamp: {best_result.timestamp}
"""
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis"""
        data = []
        for result in self.results_history:
            row = {**result.hyperparameters, **result.metrics}
            row['model_name'] = result.model_name
            row['strategy'] = result.strategy
            row['timestamp'] = result.timestamp
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_results(self, 
                    metric: str = "r_squared",
                    save_path: Optional[str] = None):
        """
        Plot hyperparameter testing results
        
        Args:
            metric: Metric to plot
            save_path: Path to save figure
        """
        df = self.get_results_dataframe()
        
        if df.empty:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Metric over iterations
        ax = axes[0, 0]
        ax.plot(df.index, df[metric], 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over Iterations')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of metric
        ax = axes[0, 1]
        ax.hist(df[metric], bins=15, edgecolor='black', alpha=0.7)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric.upper()}')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative best metric
        ax = axes[1, 0]
        if metric == 'r_squared':
            cumulative_best = df[metric].cummax()
        else:
            cumulative_best = df[metric].cummin()
        ax.plot(cumulative_best.index, cumulative_best, 'o-', linewidth=2, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'Best {metric.upper()}')
        ax.set_title(f'Best {metric.upper()} over Iterations')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of metrics
        ax = axes[1, 1]
        metric_cols = ['msae', 'r_squared', 'rmse', 'mae']
        metric_cols = [col for col in metric_cols if col in df.columns]
        
        # Normalize for heatmap
        metric_data = df[metric_cols].iloc[:10]  # Last 10 results
        metric_data_norm = (metric_data - metric_data.min()) / (metric_data.max() - metric_data.min())
        
        sns.heatmap(metric_data_norm.T, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalized Score'})
        ax.set_title('Normalized Metrics Heatmap (Last 10 Iterations)')
        ax.set_ylabel('Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        results_data = []
        for result in self.results_history:
            results_data.append({
                'hyperparameters': result.hyperparameters,
                'metrics': result.metrics,
                'timestamp': result.timestamp,
                'model_name': result.model_name,
                'strategy': result.strategy
            })
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.results_history = []
        for item in results_data:
            result = HyperparameterResult(
                hyperparameters=item['hyperparameters'],
                metrics=item['metrics'],
                timestamp=item['timestamp'],
                model_name=item['model_name'],
                strategy=item['strategy']
            )
            self.results_history.append(result)
        
        logger.info(f"Loaded {len(self.results_history)} results from {filepath}")
    
    def get_best_config(self, model_name: str) -> Optional[Tuple[Dict, HyperparameterResult]]:
        """Get best configuration for a model"""
        return self.best_configs.get(model_name)
    
    def get_summary_report(self) -> str:
        """Get summary report of all tests"""
        if not self.results_history:
            return "No results available"
        
        df = self.get_results_dataframe()
        
        report = f"""
Hyperparameter Testing Summary Report
{'='*70}

Total Tests: {len(self.results_history)}
Models Tested: {df['model_name'].nunique()}
Strategies Used: {', '.join(df['strategy'].unique())}

Performance Summary:
  R² - Min: {df['r_squared'].min():.6f}, Max: {df['r_squared'].max():.6f}, Mean: {df['r_squared'].mean():.6f}
  MSAE - Min: {df['msae'].min():.6f}, Max: {df['msae'].max():.6f}, Mean: {df['msae'].mean():.6f}
  RMSE - Min: {df['rmse'].min():.6f}, Max: {df['rmse'].max():.6f}, Mean: {df['rmse'].mean():.6f}
  MAE - Min: {df['mae'].min():.6f}, Max: {df['mae'].max():.6f}, Mean: {df['mae'].mean():.6f}

Best Configurations:
"""
        for model_name, (params, result) in self.best_configs.items():
            report += f"\n  {model_name}:\n"
            for param, value in params.items():
                report += f"    {param}: {value}\n"
            report += f"    R²: {result.metrics['r_squared']:.6f}\n"
        
        return report


# Convenience function for quick testing
def quick_hyperparameter_test(model_class: type,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray,
                             param_ranges: Dict[str, List],
                             n_iter: int = 20,
                             model_name: str = "Model") -> Tuple[Dict, float]:
    """
    Quick convenience function for hyperparameter testing
    
    Args:
        model_class: Model class to optimize
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        param_ranges: Dictionary of parameter ranges
        n_iter: Number of iterations
        model_name: Model name
        
    Returns:
        Tuple of (best_hyperparameters, best_r_squared)
    """
    tool = HyperparameterTestingTool()
    result = tool._run(
        strategy="random",
        model_class=model_class,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_config=param_ranges,
        model_name=model_name,
        n_iter=n_iter,
        scoring_metric="r_squared"
    )
    
    best_params, best_result = tool.best_configs[model_name]
    return best_params, best_result.metrics['r_squared']
