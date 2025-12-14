# HyperparameterTesting Tool - Summary

## Overview
A comprehensive hyperparameter optimization tool for your derivative pricing models. Supports Random Search, Grid Search, and multiple performance metrics (MSAE, R², RMSE, MAE).

## Files Created

### 1. **hyperparameter_testing.py** (594 lines)
Core implementation with:
- `PerformanceMetrics` class - Calculate MSAE, R², RMSE, MAE
- `RandomSearchOptimizer` - Random search optimization
- `GridSearchOptimizer` - Grid search optimization  
- `HyperparameterTestingTool` - CrewAI-compatible tool
- `HyperparameterResult` - Result storage class
- `quick_hyperparameter_test()` - Convenience function

### 2. **hyperparameter_examples.py** (350+ lines)
Complete examples:
- Example 1: Quick hyperparameter test
- Example 2: Random search with multiple models
- Example 3: Grid search for fine-tuning
- Example 4: Custom metric optimization (MSAE, R²)
- Example 5: Configuration comparison
- Example 6: CrewAI integration

### 3. **hyperparameter_integration.py** (300+ lines)
Integration patterns:
- Option 1: Add tool to existing meta_tuning_agent
- Option 2: Create dedicated HP tuning task
- Option 3: Multi-step workflow
- Option 4: Direct tool usage

### 4. **HYPERPARAMETER_TESTING_GUIDE.md** (400+ lines)
Complete documentation:
- Features and capabilities
- Installation and quick start
- API reference
- Parameter examples
- Best practices
- Troubleshooting

## Key Features

### Performance Metrics
✅ **MSAE** (Mean Squared Absolute Error) - Lower is better
✅ **R²** (Coefficient of Determination) - Higher is better  
✅ **RMSE** (Root Mean Squared Error) - Lower is better
✅ **MAE** (Mean Absolute Error) - Lower is better

### Optimization Strategies
✅ **Random Search** - Fast exploration of parameter space
✅ **Grid Search** - Exhaustive search of all combinations

### Result Management
✅ Save/load results from JSON
✅ Generate summary reports
✅ Visualize performance over iterations
✅ Compare multiple models
✅ Export to pandas DataFrame

## Quick Usage

### Method 1: One-line Quick Test
```python
from semai.tools import quick_hyperparameter_test
from sklearn.ensemble import RandomForestRegressor

best_params, best_r2 = quick_hyperparameter_test(
    model_class=RandomForestRegressor,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    param_ranges={
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
    },
    n_iter=20,
    model_name="RandomForest"
)
print(f"Best R²: {best_r2:.6f}")
```

### Method 2: Advanced with Tool Class
```python
from semai.tools import HyperparameterTestingTool
from sklearn.ensemble import RandomForestRegressor

tool = HyperparameterTestingTool()

# Run optimization
tool._run(
    strategy="random",
    model_class=RandomForestRegressor,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    param_config={
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20],
    },
    model_name="RandomForest",
    n_iter=30,
    scoring_metric="r_squared"
)

# Get results
best_params, best_result = tool.get_best_config("RandomForest")
print(f"Best Parameters: {best_params}")
print(f"Best Metrics: {best_result.metrics}")

# Visualize
tool.plot_results(metric='r_squared')
tool.save_results('hp_results.json')
```

### Method 3: In CrewAI Agent
```python
from crewai import Agent
from semai.tools import HyperparameterTestingTool

agent = Agent(
    role="Hyperparameter Tuning Specialist",
    goal="Find optimal hyperparameters",
    backstory="Expert in model optimization",
    tools=[HyperparameterTestingTool()],
    llm="gpt-4-turbo"
)
```

## Integration with Your Project

### Update meta_tuning_agent in crew.py:
```python
from semai.tools import HyperparameterTestingTool

@agent
def meta_tuning_agent(self) -> Agent:
    return Agent(
        config=self.agents_config['meta_tuning_agent'],
        tools=[HyperparameterTestingTool()],  # Add this
        verbose=True
    )
```

### Or use hyperparameter_integration.py:
```python
from semai.hyperparameter_integration import DirectHPTuning

# Compare multiple models
results = DirectHPTuning.compare_all_models(
    X_train, y_train, X_val, y_val
)
```

## Supported Models

Any scikit-learn compatible model:
- RandomForestRegressor
- GradientBoostingRegressor
- MLPRegressor
- Ridge, Lasso
- SVR
- XGBRegressor (with xgboost installed)
- LGBMRegressor (with lightgbm installed)

## Parameter Range Examples

### Random Forest
```python
{
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
```

### Gradient Boosting
```python
{
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
}
```

### Neural Network
```python
{
    'hidden_layer_sizes': [(100,), (100, 50), (100, 100, 50)],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128],
}
```

## Output Examples

### Console Output
```
Hyperparameter Optimization Results
======================================================================
Strategy: RANDOM
Model: RandomForest

Best Hyperparameters:
  n_estimators: 150
  max_depth: 15
  min_samples_split: 5
  min_samples_leaf: 2

Performance Metrics:
  msae: 0.123456
  r_squared: 0.876543
  rmse: 0.445678
  mae: 0.234567

Timestamp: 2024-12-10T12:34:56.789012
```

### Summary Report
```
Hyperparameter Testing Summary Report
======================================================================

Total Tests: 60
Models Tested: 2
Strategies Used: random

Performance Summary:
  R² - Min: 0.750000, Max: 0.899999, Mean: 0.825000
  MSAE - Min: 0.100000, Max: 0.300000, Mean: 0.200000
  ...
```

### DataFrame Export
```
    n_estimators  max_depth  min_samples_split    msae  r_squared    rmse     mae model_name   strategy
0             50          5                   2  0.250       0.750  0.500  0.250  RandomForest random
1            100         10                   5  0.150       0.850  0.387  0.150  RandomForest random
2            150         15                   10  0.125       0.875  0.354  0.125  RandomForest random
...
```

## Performance Comparison

| Strategy | Speed | Coverage | Best For |
|----------|-------|----------|----------|
| Random Search | Fast | Good | Large parameter spaces |
| Grid Search | Slow | Perfect | Small parameter spaces |

## Best Practices

1. ✅ Always use validation set separate from training
2. ✅ Start with Random Search for exploration
3. ✅ Use Grid Search for fine-tuning
4. ✅ Save results for reproducibility
5. ✅ Visualize results to understand parameter sensitivity
6. ✅ Scale parameters appropriately
7. ✅ Use meaningful parameter ranges based on domain knowledge

## Metrics Explained

### MSAE (Mean Squared Absolute Error)
- Penalizes larger errors more
- Always positive
- Lower is better
- Formula: mean(|y_true - y_pred|²)

### R² (Coefficient of Determination)
- Proportion of variance explained
- Range: -∞ to 1.0
- Higher is better
- 1.0 = perfect fit, 0 = average prediction, <0 = worse than average

### RMSE (Root Mean Squared Error)
- Same units as target variable
- Lower is better
- More sensitive to outliers than MAE
- Formula: sqrt(mean((y_true - y_pred)²))

### MAE (Mean Absolute Error)
- Same units as target variable
- Lower is better
- Robust to outliers
- Formula: mean(|y_true - y_pred|)

## Troubleshooting

### Model Fails to Train
**Solution**: Check data shapes and ensure hyperparameters are valid

### No Improvement in Results
**Solution**: Expand parameter ranges or increase iterations

### Memory Issues
**Solution**: Reduce iterations or use smaller validation set

### Slow Performance
**Solution**: Use Random Search, reduce feature count, or subsample data

## Next Steps

1. Run examples: `python hyperparameter_examples.py`
2. Test with your derivative pricing data
3. Integrate into your crew workflow
4. Compare different models
5. Save best configurations for production

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| hyperparameter_testing.py | 594 | Core tool implementation |
| hyperparameter_examples.py | 350+ | Usage examples |
| hyperparameter_integration.py | 300+ | Integration patterns |
| HYPERPARAMETER_TESTING_GUIDE.md | 400+ | Complete documentation |
| HYPERPARAMETER_TESTING_SUMMARY.md | This | Quick reference |

## Support

For issues or questions:
1. Check HYPERPARAMETER_TESTING_GUIDE.md
2. Review examples in hyperparameter_examples.py
3. Check integration patterns in hyperparameter_integration.py
4. Verify syntax with `get_errors()` tool
