"""
Side Quest Tools - Generate consistent, structured side quests for derivative pricing models
Provides tools for creating reproducible model benchmarking and validation tasks
"""

from crewai.tools import BaseTool
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
from datetime import datetime
from pathlib import Path
import hashlib


class QuestTemplate(BaseModel):
    """Template for a consistent side quest"""
    quest_id: str = Field(..., description="Unique quest identifier (auto-generated)")
    quest_name: str = Field(..., description="Human-readable quest name")
    quest_type: str = Field(..., description="Type: benchmark, validation, calibration, stress_test")
    description: str = Field(..., description="Quest objective and description")
    model_scope: List[str] = Field(..., description="Which models to test: neural_network_sde, nnlv, sdenn, 2d_nn, ann, marl")
    data_params: Dict[str, Any] = Field(..., description="Data parameters (S, K, T, r, sigma ranges)")
    success_criteria: List[str] = Field(..., description="Metrics to achieve or conditions to meet")
    expected_output: str = Field(..., description="What constitutes quest completion")
    difficulty: str = Field(..., description="Easy, Medium, Hard")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CreateQuestInput(BaseModel):
    """Input for creating a side quest"""
    quest_name: str = Field(..., description="Name of the quest")
    quest_type: str = Field(..., description="benchmark|validation|calibration|stress_test")
    description: str = Field(..., description="Quest objective")
    models: List[str] = Field(..., description="Models to test")
    success_criteria: List[str] = Field(..., description="Success metrics")
    difficulty: str = Field(default="Medium", description="Easy|Medium|Hard")
    spot_price_range: tuple = Field(default=(80, 120), description="Min and max spot prices")
    strike_multipliers: List[float] = Field(default=[0.9, 0.95, 1.0, 1.05, 1.10], description="Strike as % of spot")
    time_to_maturity_days: List[int] = Field(default=[30, 60, 90, 180], description="Days to expiry")
    volatility_range: tuple = Field(default=(0.1, 0.5), description="Min and max volatility")
    samples: int = Field(default=100, description="Number of samples to test")


class QuestTemplateInput(BaseModel):
    """Input for creating a quest template"""
    quest_name: str
    quest_type: str
    description: str
    models: List[str]


class CreateSideQuestTool(BaseTool):
    """Creates a consistent, structured side quest for model evaluation"""
    name: str = "Create Side Quest"
    description: str = (
        "Generates a structured side quest for testing derivative pricing models. "
        "Side quests are reproducible benchmarking tasks with clear success criteria, "
        "standardized data parameters, and expected outputs. Supports multiple model types "
        "and difficulty levels."
    )
    args_schema: Type[BaseModel] = CreateQuestInput

    def _run(self, 
             quest_name: str,
             quest_type: str,
             description: str,
             models: List[str],
             success_criteria: List[str],
             difficulty: str = "Medium",
             spot_price_range: tuple = (80, 120),
             strike_multipliers: List[float] = [0.9, 0.95, 1.0, 1.05, 1.10],
             time_to_maturity_days: List[int] = [30, 60, 90, 180],
             volatility_range: tuple = (0.1, 0.5),
             samples: int = 100) -> str:
        """Create a side quest with standardized parameters"""
        
        # Validate inputs
        valid_types = ['benchmark', 'validation', 'calibration', 'stress_test']
        if quest_type.lower() not in valid_types:
            return f"Error: quest_type must be one of {valid_types}"
        
        valid_models = ['neural_network_sde', 'nnlv', 'sdenn', '2d_nn', 'ann', 'marl']
        invalid_models = [m for m in models if m not in valid_models]
        if invalid_models:
            return f"Error: Invalid models {invalid_models}. Valid: {valid_models}"
        
        valid_difficulties = ['Easy', 'Medium', 'Hard']
        if difficulty not in valid_difficulties:
            return f"Error: difficulty must be one of {valid_difficulties}"
        
        # Generate quest ID
        quest_hash = hashlib.md5(
            f"{quest_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        quest_id = f"QUEST_{quest_type.upper()}_{quest_hash}"
        
        # Build data parameters
        data_params = {
            'spot_price_range': spot_price_range,
            'strike_multipliers': strike_multipliers,
            'time_to_maturity_days': time_to_maturity_days,
            'volatility_range': volatility_range,
            'samples': samples,
            'risk_free_rate': 0.02,
            'dividend_yield': 0.01
        }
        
        quest = QuestTemplate(
            quest_id=quest_id,
            quest_name=quest_name,
            quest_type=quest_type.lower(),
            description=description,
            model_scope=models,
            data_params=data_params,
            success_criteria=success_criteria,
            expected_output=self._generate_expected_output(quest_type, models),
            difficulty=difficulty
        )
        
        return json.dumps(quest.dict(), indent=2)
    
    def _generate_expected_output(self, quest_type: str, models: List[str]) -> str:
        """Generate expected output description based on quest type"""
        outputs = {
            'benchmark': f"Performance metrics (MSE, MAE, R²) for {', '.join(models)} on standardized dataset",
            'validation': f"Validation report showing model accuracy and calibration quality for {', '.join(models)}",
            'calibration': f"Calibrated volatility surfaces and parameter vectors for {', '.join(models)}",
            'stress_test': f"Model robustness analysis under extreme market conditions for {', '.join(models)}"
        }
        return outputs.get(quest_type.lower(), "Model evaluation results")


class GenerateQuestSetInput(BaseModel):
    """Input for generating a set of related quests"""
    theme: str = Field(..., description="Theme: market_conditions, model_capabilities, edge_cases, performance")
    num_quests: int = Field(default=5, description="Number of quests to generate")
    difficulty_distribution: Optional[Dict[str, int]] = Field(
        default=None, 
        description="Distribution: {'Easy': 2, 'Medium': 2, 'Hard': 1}"
    )
    target_models: List[str] = Field(default=['neural_network_sde', 'nnlv', 'sdenn'], description="Models to test")


class GenerateQuestSetTool(BaseTool):
    """Generates a thematic set of related side quests"""
    name: str = "Generate Quest Set"
    description: str = (
        "Creates a themed collection of related side quests for comprehensive model evaluation. "
        "Generates multiple quests with complementary objectives, standardized difficulty levels, "
        "and varied parameters to ensure thorough testing coverage."
    )
    args_schema: Type[BaseModel] = GenerateQuestSetInput

    def _run(self,
             theme: str,
             num_quests: int = 5,
             difficulty_distribution: Optional[Dict[str, int]] = None,
             target_models: List[str] = ['neural_network_sde', 'nnlv', 'sdenn']) -> str:
        """Generate a set of related side quests"""
        
        valid_themes = ['market_conditions', 'model_capabilities', 'edge_cases', 'performance']
        if theme.lower() not in valid_themes:
            return f"Error: theme must be one of {valid_themes}"
        
        if difficulty_distribution is None:
            # Default distribution
            if num_quests <= 3:
                difficulty_distribution = {'Easy': 1, 'Medium': 1, 'Hard': 1}
            elif num_quests <= 5:
                difficulty_distribution = {'Easy': 2, 'Medium': 2, 'Hard': 1}
            else:
                difficulty_distribution = {'Easy': 2, 'Medium': 3, 'Hard': 2}
        
        quest_sets = {
            'market_conditions': self._generate_market_conditions_quests(target_models, difficulty_distribution),
            'model_capabilities': self._generate_model_capabilities_quests(target_models, difficulty_distribution),
            'edge_cases': self._generate_edge_cases_quests(target_models, difficulty_distribution),
            'performance': self._generate_performance_quests(target_models, difficulty_distribution)
        }
        
        quests = quest_sets[theme.lower()][:num_quests]
        
        summary = {
            'theme': theme,
            'num_quests': len(quests),
            'quests': quests,
            'created_at': datetime.now().isoformat()
        }
        
        return json.dumps(summary, indent=2)
    
    def _generate_market_conditions_quests(self, models: List[str], dist: Dict[str, int]) -> List[Dict]:
        """Generate market conditions themed quests"""
        return [
            {
                'name': 'Bull Market Pricing',
                'type': 'validation',
                'description': 'Test model pricing under rising spot prices',
                'models': models,
                'difficulty': 'Easy',
                'data_params': {'spot_price_range': (100, 150), 'volatility_range': (0.1, 0.2)}
            },
            {
                'name': 'High Volatility Crisis',
                'type': 'stress_test',
                'description': 'Test model robustness with high volatility spikes',
                'models': models,
                'difficulty': 'Hard',
                'data_params': {'spot_price_range': (80, 120), 'volatility_range': (0.4, 0.8)}
            },
            {
                'name': 'Sideways Market',
                'type': 'benchmark',
                'description': 'Benchmark pricing accuracy in stable markets',
                'models': models,
                'difficulty': 'Medium',
                'data_params': {'spot_price_range': (95, 105), 'volatility_range': (0.15, 0.25)}
            },
            {
                'name': 'Volatility Collapse',
                'type': 'calibration',
                'description': 'Calibrate models as volatility drops sharply',
                'models': models,
                'difficulty': 'Hard',
                'data_params': {'spot_price_range': (80, 120), 'volatility_range': (0.05, 0.15)}
            }
        ]
    
    def _generate_model_capabilities_quests(self, models: List[str], dist: Dict[str, int]) -> List[Dict]:
        """Generate model capabilities quests"""
        return [
            {
                'name': 'Deep ITM Options',
                'type': 'validation',
                'description': 'Test pricing of deeply in-the-money options',
                'models': models,
                'difficulty': 'Easy',
                'data_params': {'strike_multipliers': [0.5, 0.6, 0.7]}
            },
            {
                'name': 'Deep OTM Options',
                'type': 'validation',
                'description': 'Test pricing of deeply out-of-money options',
                'models': models,
                'difficulty': 'Medium',
                'data_params': {'strike_multipliers': [1.3, 1.4, 1.5]}
            },
            {
                'name': 'Short Expiry Race',
                'type': 'benchmark',
                'description': 'Test theta decay pricing near expiration',
                'models': models,
                'difficulty': 'Hard',
                'data_params': {'time_to_maturity_days': [1, 3, 5, 7]}
            },
            {
                'name': 'Greeks Accuracy',
                'type': 'calibration',
                'description': 'Validate delta, gamma, vega calculations',
                'models': models,
                'difficulty': 'Hard',
                'data_params': {'samples': 200}
            }
        ]
    
    def _generate_edge_cases_quests(self, models: List[str], dist: Dict[str, int]) -> List[Dict]:
        """Generate edge case quests"""
        return [
            {
                'name': 'Extreme Low Volatility',
                'type': 'stress_test',
                'description': 'Test with volatility near zero',
                'models': models,
                'difficulty': 'Easy',
                'data_params': {'volatility_range': (0.01, 0.05)}
            },
            {
                'name': 'Extreme High Volatility',
                'type': 'stress_test',
                'description': 'Test with volatility > 100%',
                'models': models,
                'difficulty': 'Hard',
                'data_params': {'volatility_range': (1.0, 2.0)}
            },
            {
                'name': 'Near Expiration Edge',
                'type': 'benchmark',
                'description': 'Pricing behavior very close to expiry',
                'models': models,
                'difficulty': 'Medium',
                'data_params': {'time_to_maturity_days': [0.1, 0.5, 1]}
            }
        ]
    
    def _generate_performance_quests(self, models: List[str], dist: Dict[str, int]) -> List[Dict]:
        """Generate performance quests"""
        return [
            {
                'name': 'Calibration Speed',
                'type': 'benchmark',
                'description': 'Measure model training/calibration time',
                'models': models,
                'difficulty': 'Easy',
                'data_params': {'samples': 1000}
            },
            {
                'name': 'Large Dataset Efficiency',
                'type': 'benchmark',
                'description': 'Test with 10k+ pricing samples',
                'models': models,
                'difficulty': 'Medium',
                'data_params': {'samples': 10000}
            },
            {
                'name': 'Real-Time Inference',
                'type': 'stress_test',
                'description': 'Prediction latency under load',
                'models': models,
                'difficulty': 'Hard',
                'data_params': {'samples': 100000}
            }
        ]


class QuestProgressInput(BaseModel):
    """Input for tracking quest progress"""
    quest_id: str = Field(..., description="Quest ID to track")
    model_name: str = Field(..., description="Model being tested")
    status: str = Field(..., description="in_progress|completed|failed")
    metrics: Dict[str, float] = Field(..., description="Performance metrics achieved")
    notes: Optional[str] = Field(None, description="Additional notes")


class TrackQuestProgressTool(BaseTool):
    """Tracks side quest completion and results"""
    name: str = "Track Quest Progress"
    description: str = (
        "Logs progress on a side quest, records metrics and status. "
        "Used to track which models complete which quests and their performance results."
    )
    args_schema: Type[BaseModel] = QuestProgressInput

    def _run(self,
             quest_id: str,
             model_name: str,
             status: str,
             metrics: Dict[str, float],
             notes: Optional[str] = None) -> str:
        """Track quest progress"""
        
        valid_statuses = ['in_progress', 'completed', 'failed']
        if status not in valid_statuses:
            return f"Error: status must be one of {valid_statuses}"
        
        progress_record = {
            'quest_id': quest_id,
            'model_name': model_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'notes': notes
        }
        
        return json.dumps(progress_record, indent=2)


class QuestLeaderboardInput(BaseModel):
    """Input for generating leaderboard"""
    metric: str = Field(default='accuracy', description="Metric to rank by")
    limit: int = Field(default=10, description="Top N results")


class QuestLeaderboardTool(BaseTool):
    """Generates leaderboard of model performance across quests"""
    name: str = "Generate Quest Leaderboard"
    description: str = (
        "Creates a performance leaderboard showing which models excelled at different quest types. "
        "Useful for comparing model capabilities across benchmarks and validations."
    )
    args_schema: Type[BaseModel] = QuestLeaderboardInput

    def _run(self, metric: str = 'accuracy', limit: int = 10) -> str:
        """Generate leaderboard"""
        
        leaderboard = {
            'metric': metric,
            'limit': limit,
            'timestamp': datetime.now().isoformat(),
            'note': 'Leaderboard generation requires active quest tracking data'
        }
        
        return json.dumps(leaderboard, indent=2)
