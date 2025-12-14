"""
LLAMA Guardrail Safety Model - Usage Examples

This file demonstrates how the LLAMA guardrail safety model works
throughout the entire workflow.
"""

import numpy as np
from semai.builtin_models import BlackScholesModel, LinearRegressionModel

# ============================================================================
# Example 1: Basic Usage with Automatic Safety Checks
# ============================================================================
def example_1_basic_safety():
    """Example 1: Models automatically validate data with guardrail."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Safety Checks")
    print("="*70)
    
    # Create model (guardrail enabled by default)
    model = BlackScholesModel()
    
    # Create valid test data
    X_test = np.array([
        [100, 100, 1.0, 0.05, 0.05, 0.2],  # S, K, T, q, r, sigma
        [110, 100, 1.0, 0.05, 0.05, 0.2],
        [90,  100, 1.0, 0.05, 0.05, 0.2],
    ])
    
    y_test = np.array([10.45, 15.2, 5.8])
    
    print("\nMaking predictions with valid data...")
    y_pred = model.predict(X_test)
    print(f"✅ Predictions made successfully: {y_pred}")
    
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"✅ Evaluation complete: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")


# ============================================================================
# Example 2: Safety Alerts on Invalid Data
# ============================================================================
def example_2_safety_alerts():
    """Example 2: Guardrail detects and warns about invalid data."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Safety Alerts on Invalid Data")
    print("="*70)
    
    model = BlackScholesModel()
    
    # Create INVALID test data (negative stock price)
    X_invalid = np.array([
        [-100, 100, 1.0, 0.05, 0.05, 0.2],  # ❌ NEGATIVE stock price!
        [110, 100, 1.0, 0.05, 0.05, 0.2],
    ])
    
    print("\nAttempting prediction with INVALID data...")
    print("(Guardrail will detect issues and issue warnings)\n")
    
    try:
        y_pred = model.predict(X_invalid)
        print(f"Predictions made (with warnings): {y_pred}")
    except Exception as e:
        print(f"Error during prediction: {e}")


# ============================================================================
# Example 3: Accessing Safety Report
# ============================================================================
def example_3_safety_report():
    """Example 3: Access detailed safety report."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Accessing Safety Reports")
    print("="*70)
    
    model = LinearRegressionModel(learning_rate=0.01, iterations=100)
    
    # Create training data
    X_train = np.random.rand(100, 6) * 100
    X_train[:, 0] = np.abs(X_train[:, 0]) + 50  # Stock price > 50
    X_train[:, 1] = np.abs(X_train[:, 1]) + 50  # Strike price > 50
    y_train = np.random.rand(100) * 20 + 5
    
    print("\nTraining model...")
    model.train(X_train, y_train)
    
    # Get safety report
    report = model.guardrail.get_safety_report()
    print("\n--- Safety Report ---")
    print(f"Total checks: {report['total_checks']}")
    print(f"Passed checks: {report['passed_checks']}")
    print(f"Failed checks: {report['failed_checks']}")
    print(f"Pass rate: {report['pass_rate']:.1%}")
    print(f"Violations: {report['violations_count']}")
    
    # Print detailed report
    print("\n--- Detailed Report ---")
    model.guardrail.print_safety_report()


# ============================================================================
# Example 4: Enabling Verbose Safety Monitoring
# ============================================================================
def example_4_verbose_monitoring():
    """Example 4: Enable verbose output to see all safety checks."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Verbose Safety Monitoring")
    print("="*70)
    
    model = BlackScholesModel()
    model.guardrail.verbose = True  # Enable detailed output
    
    X_test = np.array([
        [100, 100, 1.0, 0.05, 0.05, 0.2],
        [110, 100, 1.0, 0.05, 0.05, 0.2],
    ])
    
    print("\nMaking predictions with VERBOSE guardrail output...")
    y_pred = model.predict(X_test)
    print(f"\nPredictions: {y_pred}")


# ============================================================================
# Example 5: Disabling Guardrail for Performance
# ============================================================================
def example_5_disable_guardrail():
    """Example 5: Disable guardrail if not needed (performance)."""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Disabling Guardrail")
    print("="*70)
    
    model = BlackScholesModel()
    model.enable_guardrail = False  # Disable safety checks
    
    X_test = np.random.rand(1000, 6) * 100 + 50
    
    print("\nMaking 1000 predictions WITHOUT guardrail (faster)...")
    y_pred = model.predict(X_test)
    print(f"✅ Completed. Shape: {y_pred.shape}")
    
    # Compare with guardrail enabled
    model.enable_guardrail = True
    print("\nMaking same predictions WITH guardrail (slight overhead)...")
    y_pred = model.predict(X_test)
    print(f"✅ Completed. Shape: {y_pred.shape}")


# ============================================================================
# Example 6: Custom Safety Threshold
# ============================================================================
def example_6_custom_threshold():
    """Example 6: Adjust safety threshold sensitivity."""
    
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Safety Threshold")
    print("="*70)
    
    # Strict safety (requires 95% confidence)
    model_strict = LinearRegressionModel()
    model_strict.guardrail.safety_threshold = 0.95
    print("Strict model: threshold = 0.95 (very strict)")
    
    # Lenient safety (requires 60% confidence)
    model_lenient = LinearRegressionModel()
    model_lenient.guardrail.safety_threshold = 0.60
    print("Lenient model: threshold = 0.60 (more permissive)")
    
    # Default safety (requires 80% confidence)
    model_default = LinearRegressionModel()
    print(f"Default model: threshold = {model_default.guardrail.safety_threshold}")


# ============================================================================
# Example 7: Comprehensive Training Workflow
# ============================================================================
def example_7_full_workflow():
    """Example 7: Complete training and evaluation with safety checks."""
    
    print("\n" + "="*70)
    print("EXAMPLE 7: Full Workflow with Safety")
    print("="*70)
    
    # Create model
    model = LinearRegressionModel(learning_rate=0.01, iterations=50)
    
    # Generate synthetic derivative pricing data
    np.random.seed(42)
    n_samples = 200
    
    X_train = np.random.rand(n_samples, 6) * 50 + 50
    X_train[:, 0] = np.abs(X_train[:, 0]) + 80  # Stock price: 80-130
    X_train[:, 1] = np.abs(X_train[:, 1]) + 80  # Strike price: 80-130
    X_train[:, 2] = np.abs(X_train[:, 2]) + 0.1  # Time: 0.1-0.6 years
    y_train = np.random.rand(n_samples) * 30 + 5  # Option price: 5-35
    
    X_test = X_train[:50]  # Use subset for testing
    y_test = y_train[:50]
    
    print("\n1. Training with safety monitoring...")
    model.train(X_train, y_train)
    print("   ✅ Training complete")
    
    print("\n2. Making predictions...")
    y_pred = model.predict(X_test)
    print(f"   ✅ Predictions made: {len(y_pred)} samples")
    
    print("\n3. Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"   ✅ RMSE: {metrics['rmse']:.4f}")
    print(f"   ✅ R²: {metrics['r2']:.4f}")
    print(f"   ✅ MAE: {metrics['mae']:.4f}")
    
    print("\n4. Safety report...")
    report = model.guardrail.get_safety_report()
    print(f"   ✅ Total checks: {report['total_checks']}")
    print(f"   ✅ Pass rate: {report['pass_rate']:.1%}")
    print(f"   ✅ Violations: {report['violations_count']}")


# ============================================================================
# Run All Examples
# ============================================================================
def run_all_examples():
    """Run all guardrail examples."""
    
    print("\n" + "="*70)
    print("LLAMA GUARDRAIL SAFETY MODEL - USAGE EXAMPLES")
    print("="*70)
    
    try:
        example_1_basic_safety()
        example_2_safety_alerts()
        example_3_safety_report()
        example_4_verbose_monitoring()
        example_5_disable_guardrail()
        example_6_custom_threshold()
        example_7_full_workflow()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
