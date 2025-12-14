"""
Example Side Quests - Pre-built quest templates for consistent model benchmarking
These can be loaded directly or used as templates for creating new quests
"""

QUEST_TEMPLATES = {
    'benchmark_accuracy': {
        'quest_name': 'Accuracy Benchmark',
        'quest_type': 'benchmark',
        'description': 'Measure baseline pricing accuracy across all models on standard dataset',
        'models': ['neural_network_sde', 'nnlv', 'sdenn', '2d_nn', 'ann', 'marl'],
        'difficulty': 'Easy',
        'success_criteria': [
            'MSE < 0.01',
            'MAE < 5% of option price',
            'R² > 0.95'
        ],
        'data_params': {
            'spot_price_range': (80, 120),
            'strike_multipliers': [0.9, 0.95, 1.0, 1.05, 1.10],
            'time_to_maturity_days': [30, 60, 90, 180],
            'volatility_range': (0.15, 0.35),
            'samples': 1000
        }
    },
    
    'stress_test_extreme_vol': {
        'quest_name': 'Extreme Volatility Stress Test',
        'quest_type': 'stress_test',
        'description': 'Test model robustness when volatility spikes above 100%',
        'models': ['neural_network_sde', 'nnlv', 'sdenn', '2d_nn'],
        'difficulty': 'Hard',
        'success_criteria': [
            'No numerical errors',
            'Prices remain positive',
            'Greeks are well-defined'
        ],
        'data_params': {
            'spot_price_range': (80, 120),
            'strike_multipliers': [0.9, 1.0, 1.1],
            'time_to_maturity_days': [30, 90],
            'volatility_range': (0.5, 2.0),
            'samples': 500
        }
    },
    
    'calibration_market_fit': {
        'quest_name': 'Market Data Calibration',
        'quest_type': 'calibration',
        'description': 'Calibrate models to match real market option prices',
        'models': ['nnlv', 'sdenn', 'marl'],
        'difficulty': 'Hard',
        'success_criteria': [
            'Calibration error < 2%',
            'Implied vol surface smooth',
            'No arbitrage violations'
        ],
        'data_params': {
            'spot_price_range': (95, 105),
            'strike_multipliers': [0.95, 0.98, 1.0, 1.02, 1.05],
            'time_to_maturity_days': [7, 30, 60, 90],
            'volatility_range': (0.15, 0.30),
            'samples': 2000
        }
    },
    
    'validation_greeks': {
        'quest_name': 'Greeks Computation Validation',
        'quest_type': 'validation',
        'description': 'Validate delta, gamma, vega, theta, rho calculations against analytical solutions',
        'models': ['neural_network_sde', 'nnlv', 'ann'],
        'difficulty': 'Medium',
        'success_criteria': [
            'Delta error < 1%',
            'Gamma error < 5%',
            'Vega error < 5%'
        ],
        'data_params': {
            'spot_price_range': (90, 110),
            'strike_multipliers': [0.95, 1.0, 1.05],
            'time_to_maturity_days': [30, 60, 90],
            'volatility_range': (0.15, 0.35),
            'samples': 300
        }
    },
    
    'benchmark_speed': {
        'quest_name': 'Calibration Speed Competition',
        'quest_type': 'benchmark',
        'description': 'Compare model training/calibration speed on large dataset',
        'models': ['neural_network_sde', 'sdenn', '2d_nn', 'ann', 'marl'],
        'difficulty': 'Medium',
        'success_criteria': [
            'Calibration time < 5 minutes',
            'Convergence achieved',
            'Batch processing stable'
        ],
        'data_params': {
            'spot_price_range': (80, 120),
            'strike_multipliers': [0.9, 1.0, 1.1],
            'time_to_maturity_days': [30, 60, 90, 180],
            'volatility_range': (0.1, 0.5),
            'samples': 5000
        }
    },
    
    'edge_case_deep_itm': {
        'quest_name': 'Deep In-The-Money Options',
        'quest_type': 'validation',
        'description': 'Test pricing accuracy for deeply ITM options (intrinsic value dominates)',
        'models': ['neural_network_sde', 'nnlv', 'sdenn', 'ann'],
        'difficulty': 'Easy',
        'success_criteria': [
            'Price ≈ intrinsic value',
            'Theta = 0 (no time value)',
            'Delta ≈ 1'
        ],
        'data_params': {
            'spot_price_range': (100, 200),
            'strike_multipliers': [0.5, 0.6, 0.7],
            'time_to_maturity_days': [30, 60],
            'volatility_range': (0.15, 0.35),
            'samples': 200
        }
    },
    
    'edge_case_deep_otm': {
        'quest_name': 'Deep Out-Of-The-Money Options',
        'quest_type': 'validation',
        'description': 'Test pricing accuracy for deeply OTM options (small probabilities)',
        'models': ['neural_network_sde', 'nnlv', 'sdenn', '2d_nn'],
        'difficulty': 'Hard',
        'success_criteria': [
            'Prices remain positive',
            'Monotonic in strike',
            'No artificial boundaries'
        ],
        'data_params': {
            'spot_price_range': (80, 120),
            'strike_multipliers': [1.3, 1.4, 1.5, 1.6],
            'time_to_maturity_days': [30, 90],
            'volatility_range': (0.15, 0.35),
            'samples': 300
        }
    },
    
    'edge_case_near_expiry': {
        'quest_name': 'Near Expiration Pricing',
        'quest_type': 'stress_test',
        'description': 'Test model behavior very close to expiration (theta acceleration)',
        'models': ['neural_network_sde', 'nnlv', 'sdenn', 'ann'],
        'difficulty': 'Hard',
        'success_criteria': [
            'Converges to intrinsic value',
            'Theta matches Black-Scholes',
            'Stable numerical behavior'
        ],
        'data_params': {
            'spot_price_range': (90, 110),
            'strike_multipliers': [0.95, 1.0, 1.05],
            'time_to_maturity_days': [0.1, 0.5, 1, 2, 7],
            'volatility_range': (0.15, 0.35),
            'samples': 500
        }
    },
    
    'model_comparison_smile': {
        'quest_name': 'Volatility Smile Capture',
        'quest_type': 'validation',
        'description': 'Test ability to capture volatility smile across strikes',
        'models': ['nnlv', 'sdenn', '2d_nn', 'marl'],
        'difficulty': 'Hard',
        'success_criteria': [
            'Smile shape recognized',
            'Volatility varies with strike',
            'Consistent with market'
        ],
        'data_params': {
            'spot_price_range': (95, 105),
            'strike_multipliers': [0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10],
            'time_to_maturity_days': [30, 90],
            'volatility_range': (0.15, 0.35),
            'samples': 1000
        }
    },
    
    'model_comparison_term_structure': {
        'quest_name': 'Term Structure Consistency',
        'quest_type': 'calibration',
        'description': 'Test consistency of pricing across different maturities',
        'models': ['nnlv', 'sdenn', 'marl'],
        'difficulty': 'Medium',
        'success_criteria': [
            'No calendar arbitrage',
            'Smooth term structure',
            'Consistent carry'
        ],
        'data_params': {
            'spot_price_range': (95, 105),
            'strike_multipliers': [1.0],
            'time_to_maturity_days': [7, 14, 30, 60, 90, 180, 365],
            'volatility_range': (0.15, 0.35),
            'samples': 300
        }
    }
}


def get_quest_template(template_name: str) -> dict:
    """Get a pre-built quest template"""
    if template_name not in QUEST_TEMPLATES:
        available = list(QUEST_TEMPLATES.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")
    
    return QUEST_TEMPLATES[template_name].copy()


def list_available_templates() -> list:
    """List all available quest templates"""
    return list(QUEST_TEMPLATES.keys())
