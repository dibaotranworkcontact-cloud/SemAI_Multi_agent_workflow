# HyperparameterTesting Tool Documentation

## Overview
The `HyperparameterTesting` tool provides comprehensive hyperparameter optimization capabilities for machine learning models using Random Search, Grid Search, and multiple performance metrics.

## Features

### 1. Optimization Strategies

#### Random Search
- Randomly samples from parameter distributions
- Fast exploration of parameter space
- Good for high-dimensional parameter spaces
- Configurable number of iterations

#### Grid Search
- Exhaustively searches all parameter combinations
- Comprehensive but computationally expensive
- Better for smaller parameter spaces
- Guarantees exploration of all combinations

### 2. Performance Metrics

The tool evaluates models using multiple metrics:

- **MSAE (Mean Squared Absolute Error)**
  - Formula: mean(|y_true - y_pred|²)
  - Lower is better
  - Emphasizes larger errors

- **R² (Coefficient of Determination)**
  - Formula: 1 - (SS_res / SS_tot)
  - Higher is better (max 1.0)
  - Proportion of variance explained

- **RMSE (Root Mean Squared Error)**
  - Formula: sqrt(mean((y_true - y_pred)²))
  - Lower is better
  - In same units as target

- **MAE (Mean Absolute Error)**
  - Formula: mean(|y_true - y_pred|)
  - Lower is better
  - Robust to outliers

### 3. Result Management

- Store results in memory or JSON
- Generate summary reports
- Visualize performance across iterations
- Compare multiple models
- Export results for analysis

## Installation

The tool is already installed and available in your project:

```python
from semai.tools import HyperparameterTestingTool
```

## Quick Start

### Basic Usage

```python
from semai.tools import quick_hyperparameter_test
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=1000, n_features=20, noise=10)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define parameter ranges
param_ranges = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
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

print(f"Best Parameters: {best_params}")
print(f"Best R²: {best_r2:.6f}")
```

### Advanced Usage with Tool Class

```python
from semai.tools import HyperparameterTestingTool
from sklearn.ensemble import RandomForestRegressor

# Initialize tool
tool = HyperparameterTestingTool()

# Run Random Search
result = tool._run(
    strategy="random",
    model_class=RandomForestRegressor,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    param_config={
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
    },
    model_name="RandomForest",
    n_iter=30,
    scoring_metric="r_squared"
)

print(result)
```

### Grid Search

```python
# Define parameter grid (smaller than random search)
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
}

# Run Grid Search
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
```

## API Reference

### HyperparameterTestingTool

#### Methods

##### `_run(strategy, model_class, X_train, y_train, X_val, y_val, param_config, model_name, n_iter, scoring_metric)`

Run hyperparameter optimization.

**Parameters:**
- `strategy` (str): 'random' or 'grid'
- `model_class` (type): Model class to instantiate
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training targets
- `X_val` (np.ndarray): Validation features
- `y_val` (np.ndarray): Validation targets
- `param_config` (dict): Parameter configuration
- `model_name` (str): Name of model being optimized
- `n_iter` (int): Number of iterations (for random search)
- `scoring_metric` (str): 'r_squared' or 'msae'

**Returns:**
- str: Result summary

##### `get_results_dataframe()`

Convert all results to a pandas DataFrame.

**Returns:**
- pd.DataFrame: Results with hyperparameters and metrics

##### `plot_results(metric, save_path)`

Generate visualization of optimization results.

**Parameters:**
- `metric` (str): Metric to plot ('r_squared', 'msae', 'rmse', 'mae')
- `save_path` (str, optional): Path to save figure

##### `save_results(filepath)`

Save results to JSON file.

**Parameters:**
- `filepath` (str): Path to save JSON file

##### `load_results(filepath)`

Load results from JSON file.

**Parameters:**
- `filepath` (str): Path to JSON file

##### `get_best_config(model_name)`

Get best configuration for a model.

**Parameters:**
- `model_name` (str): Name of the model

**Returns:**
- Tuple[Dict, HyperparameterResult]: (best_hyperparameters, best_result)

##### `get_summary_report()`

Generate summary report of all tests.

**Returns:**
- str: Formatted summary report

### PerformanceMetrics

Static utility class for calculating metrics.

#### Methods

##### `calculate_msae(y_true, y_pred)`
##### `calculate_r_squared(y_true, y_pred)`
##### `calculate_rmse(y_true, y_pred)`
##### `calculate_mae(y_true, y_pred)`
##### `get_all_metrics(y_true, y_pred)`

## Examples

### Example 1: Quick Test

```python
from semai.tools import quick_hyperparameter_test
from sklearn.ensemble import RandomForestRegressor

best_params, best_r2 = quick_hyperparameter_test(
    model_class=RandomForestRegressor,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    param_ranges={
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
    },
    n_iter=20,
    model_name="RF"
)
```

### Example 2: Multiple Models Comparison

```python
tool = HyperparameterTestingTool()

# Test Random Forest
tool._run(
    strategy="random",
    model_class=RandomForestRegressor,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    param_config={'n_estimators': [50, 100, 200]},
    model_name="RandomForest",
    n_iter=20,
    scoring_metric="r_squared"
)

# Test Gradient Boosting
tool._run(
    strategy="random",
    model_class=GradientBoostingRegressor,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    param_config={'n_estimators': [50, 100, 200]},
    model_name="GradientBoosting",
    n_iter=20,
    scoring_metric="r_squared"
)

# Compare results
df = tool.get_results_dataframe()
print(df.groupby('model_name')[['r_squared', 'msae']].mean())
```

### Example 3: Optimize for Different Metrics

```python
# Optimize for R² (higher is better)
tool._run(..., scoring_metric="r_squared")

# Optimize for MSAE (lower is better)
tool._run(..., scoring_metric="msae")

# Optimize for RMSE (lower is better)
# Note: Can view in results dataframe
```

### Example 4: Analysis and Visualization

```python
# Get results as DataFrame
df = tool.get_results_dataframe()

# Filter best results
best_rf = df[df['model_name'] == 'RandomForest'].nlargest(1, 'r_squared')
print(best_rf)

# Plot results
tool.plot_results(metric='r_squared', save_path='results.png')

# Get summary report
print(tool.get_summary_report())

# Save for later analysis
tool.save_results('hp_results.json')
```

## Integration with CrewAI

```python
from crewai import Agent
from semai.tools import HyperparameterTestingTool

agent = Agent(
    role="Hyperparameter Tuning Specialist",
    goal="Find optimal hyperparameters for ML models",
    backstory="Expert in model optimization",
    tools=[HyperparameterTestingTool()],
    llm="gpt-4-turbo"
)
```

## Parameter Range Examples

### Random Forest

```python
{
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```

### Gradient Boosting

```python
{
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}
```

### Neural Network

```python
{
    'hidden_layer_sizes': [(100,), (100, 50), (100, 100, 50)],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128],
    'alpha': [0.0001, 0.001, 0.01]
}
```

## Best Practices

1. **Use Validation Split**: Always use a separate validation set for hyperparameter tuning
2. **Choose Metrics Wisely**: Select metrics that align with your business goals
3. **Start with Random Search**: Quick exploration before fine-tuning with Grid Search
4. **Save Results**: Always save results for reproducibility and analysis
5. **Visualize Results**: Use plots to understand parameter-performance relationships
6. **Scale Parameters**: Ensure parameters are properly scaled (especially learning rates)
7. **Use Meaningful Ranges**: Base ranges on domain knowledge and previous experiments

## Troubleshooting

### Model Fails to Train
- Check if data shapes match expected input
- Ensure hyperparameters are valid for the model
- Try with simpler model first

### Results Show No Improvement
- Expand parameter ranges
- Increase number of iterations
- Check if validation set is representative

### Memory Issues with Large Datasets
- Reduce number of iterations
- Use smaller validation set
- Consider subsampling training data

## Performance Tips

1. **Parallel Processing**: Consider parallel optimization for faster results
2. **Early Stopping**: Stop if no improvement after N iterations
3. **Feature Selection**: Reduce features to speed up optimization
4. **Smaller Validation Set**: Use subset for faster feedback
5. **Coarse to Fine**: Start coarse, then refine around best parameters

## Files

- **hyperparameter_testing.py**: Main tool implementation
- **hyperparameter_examples.py**: Example usage and demonstrations

## References

- scikit-learn documentation on hyperparameter optimization
- Bergstra & Bengio (2012): Random Search for Hyper-Parameter Optimization
- Hyperopt: Hyperparameter Optimization
