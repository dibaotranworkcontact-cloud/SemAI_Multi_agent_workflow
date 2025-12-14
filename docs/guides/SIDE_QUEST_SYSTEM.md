## Side Quest System - Consistent Model Benchmarking Framework

### Overview

The **Side Quest System** enables consistent, reproducible benchmarking of derivative pricing models through standardized quest templates and a central registry.

### What Are Side Quests?

Side quests are **structured benchmarking tasks** that:
- Have standardized parameters and success criteria
- Test specific model capabilities or market scenarios
- Track performance metrics consistently
- Build toward comprehensive model evaluation
- Support comparison across models and time periods

### Components

#### 1. **CrewAI Tools** (`tools/side_quest_tools.py`)

**CreateSideQuestTool**
- Creates individual quests with standardized parameters
- Generates unique quest IDs
- Validates model types and difficulty levels
- Produces JSON specification

Example:
```python
quest_tool = CreateSideQuestTool()
result = quest_tool._run(
    quest_name='Accuracy Test',
    quest_type='benchmark',
    description='Test baseline pricing accuracy',
    models=['neural_network_sde', 'nnlv', 'sdenn'],
    success_criteria=['MSE < 0.01', 'R² > 0.95'],
    difficulty='Easy'
)
```

**GenerateQuestSetTool**
- Creates themed collections of related quests
- Supports themes: market_conditions, model_capabilities, edge_cases, performance
- Auto-generates difficulty distribution
- Produces multiple complementary quests

Example:
```python
set_tool = GenerateQuestSetTool()
result = set_tool._run(
    theme='edge_cases',
    num_quests=5,
    target_models=['neural_network_sde', 'nnlv']
)
```

**TrackQuestProgressTool**
- Logs model performance on quests
- Records metrics and status
- Timestamps all entries
- Enables historical tracking

Example:
```python
progress_tool = TrackQuestProgressTool()
progress_tool._run(
    quest_id='QUEST_BENCHMARK_abc123',
    model_name='neural_network_sde',
    status='completed',
    metrics={'mse': 0.008, 'r2': 0.96}
)
```

**QuestLeaderboardTool**
- Ranks models by performance metric
- Supports custom metrics (accuracy, speed, stability, etc.)
- Generates comparative leaderboards

#### 2. **Quest Registry** (`quest_registry.py`)

Central repository for quest definitions and results.

**Features:**
- Persistent storage of quest definitions
- Result logging for all model attempts
- Quest set management
- Performance leaderboards
- Statistics and reporting

**Usage:**
```python
from semai.quest_registry import get_quest_registry

registry = get_quest_registry()

# Register a quest
quest_id = registry.register_quest(quest_data)

# Log results
result_id = registry.log_result(
    quest_id=quest_id,
    model_name='model_name',
    result_data={'status': 'completed', 'metrics': {...}}
)

# Generate leaderboard
leaderboard = registry.generate_leaderboard(metric='accuracy', limit=10)

# Get stats
stats = registry.get_stats()
```

#### 3. **Example Quest Templates** (`example_quests.py`)

10 pre-built quest templates for quick reference:

1. **benchmark_accuracy** - Baseline pricing accuracy
2. **stress_test_extreme_vol** - High volatility robustness
3. **calibration_market_fit** - Real market data calibration
4. **validation_greeks** - Greeks computation accuracy
5. **benchmark_speed** - Calibration speed comparison
6. **edge_case_deep_itm** - In-the-money option pricing
7. **edge_case_deep_otm** - Out-of-money option pricing
8. **edge_case_near_expiry** - Near-expiration behavior
9. **model_comparison_smile** - Volatility smile capture
10. **model_comparison_term_structure** - Term structure consistency

### Quest Types

**Benchmark**
- Measure performance on standardized dataset
- Compare models on equal footing
- Typical success metric: MSE, R², accuracy

**Validation**
- Verify model against known solutions
- Check Greeks and analytical formulas
- Validate market assumptions

**Calibration**
- Fit model to real or synthetic market data
- Test parameter recovery
- Verify surface smoothness

**Stress Test**
- Test robustness under extreme conditions
- Check numerical stability
- Verify limits and boundaries

### Difficulty Levels

**Easy**
- Standard market conditions
- Clear solutions
- Limited parameter ranges
- Few edge cases

**Medium**
- Varied market scenarios
- Moderate computational complexity
- Some parameter extremes
- Some edge case handling

**Hard**
- Extreme market conditions
- Complex interactions
- Full parameter ranges
- Edge case dominance
- May require special handling

### Quest Themes

**market_conditions**
- Bull/bear/sideways markets
- Volatility spikes and crashes
- Different vol regimes

**model_capabilities**
- Deep ITM/OTM options
- Short expiry pricing
- Greeks calculation

**edge_cases**
- Extreme volatility
- Near expiration
- Boundary conditions

**performance**
- Calibration speed
- Large datasets
- Real-time inference

### Directory Structure

```
.quests/
├── definitions/
│   └── QUEST_*.json          # Quest specifications
├── results/
│   └── QUEST_*_model_*.json  # Individual result logs
└── sets/
    └── SET_*.json            # Quest set collections
```

### Workflow Example

```python
from semai.tools.side_quest_tools import CreateSideQuestTool, TrackQuestProgressTool
from semai.quest_registry import get_quest_registry
from semai.example_quests import get_quest_template

registry = get_quest_registry()

# Step 1: Load or create a quest
template = get_quest_template('benchmark_accuracy')
quest_data = {
    'quest_id': 'QUEST_ACCURACY_001',
    **template
}

# Step 2: Register quest
registry.register_quest(quest_data)

# Step 3: Run models
models = ['neural_network_sde', 'nnlv', 'sdenn']
for model_name in models:
    # ... train and evaluate model ...
    metrics = model.evaluate(test_data)
    
    # Step 4: Log results
    registry.log_result(
        quest_id=quest_data['quest_id'],
        model_name=model_name,
        result_data={
            'status': 'completed',
            'metrics': metrics
        }
    )

# Step 5: Generate leaderboard
leaderboard = registry.generate_leaderboard(metric='r2')
print(leaderboard)
```

### Integration with CrewAI

Add tools to agents:

```python
from semai.tools.side_quest_tools import (
    CreateSideQuestTool,
    GenerateQuestSetTool,
    TrackQuestProgressTool,
    QuestLeaderboardTool
)

# Add to agent tools
tools = [
    CreateSideQuestTool(),
    GenerateQuestSetTool(),
    TrackQuestProgressTool(),
    QuestLeaderboardTool()
]
```

### Key Benefits

✅ **Consistency** - Standardized parameters across all models  
✅ **Reproducibility** - Same quest conditions produce same results  
✅ **Comparison** - Fair side-by-side model evaluation  
✅ **Progress Tracking** - Historical record of model improvements  
✅ **Leaderboards** - Automatic ranking and comparison  
✅ **Scalability** - Add new quests and models without restructuring  
✅ **Themes** - Organized evaluation across different scenarios  

### Extending the System

**Add new quest template:**
```python
# In example_quests.py
QUEST_TEMPLATES['custom_template'] = {
    'quest_name': 'Custom Test',
    'quest_type': 'benchmark',
    # ... parameters ...
}
```

**Add new theme in GenerateQuestSetTool:**
```python
def _generate_custom_theme_quests(self, models, dist):
    return [
        {'name': 'Custom Quest 1', ...},
        {'name': 'Custom Quest 2', ...}
    ]
```

**Custom metrics:**
```python
registry.generate_leaderboard(metric='custom_metric', limit=10)
```

### Status

✅ Complete and tested
✅ 10 example templates ready
✅ Full registry functionality
✅ Integration ready for CrewAI agents
✅ JSON-based persistence for auditability

