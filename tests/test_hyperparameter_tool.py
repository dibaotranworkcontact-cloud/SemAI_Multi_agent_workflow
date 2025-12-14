"""
Quick Test of HyperparameterTesting Tool
Verifies tool is working correctly
"""

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    
    try:
        from semai.tools import (
            HyperparameterTestingTool,
            RandomSearchOptimizer,
            GridSearchOptimizer,
            PerformanceMetrics,
            quick_hyperparameter_test
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_performance_metrics():
    """Test performance metrics calculation"""
    print("\nTesting performance metrics...")
    
    import numpy as np
    from semai.tools import PerformanceMetrics
    
    # Create test data
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    try:
        msae = PerformanceMetrics.calculate_msae(y_true, y_pred)
        r2 = PerformanceMetrics.calculate_r_squared(y_true, y_pred)
        rmse = PerformanceMetrics.calculate_rmse(y_true, y_pred)
        mae = PerformanceMetrics.calculate_mae(y_true, y_pred)
        
        print(f"✅ MSAE: {msae:.6f}")
        print(f"✅ R²: {r2:.6f}")
        print(f"✅ RMSE: {rmse:.6f}")
        print(f"✅ MAE: {mae:.6f}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_tool_initialization():
    """Test tool initialization"""
    print("\nTesting tool initialization...")
    
    try:
        from semai.tools import HyperparameterTestingTool
        
        tool = HyperparameterTestingTool()
        print(f"✅ Tool created: {tool.name}")
        print(f"✅ Description: {tool.description[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_quick_hyperparameter_test():
    """Test quick hyperparameter test function"""
    print("\nTesting quick_hyperparameter_test function...")
    
    try:
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from semai.tools import quick_hyperparameter_test
        
        # Generate small test data
        X, y = make_regression(n_samples=100, n_features=5, noise=1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Run quick test
        best_params, best_r2 = quick_hyperparameter_test(
            model_class=RandomForestRegressor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_ranges={
                'n_estimators': [10, 20],
                'max_depth': [2, 3],
            },
            n_iter=5,
            model_name="TestModel"
        )
        
        print(f"✅ Best Parameters: {best_params}")
        print(f"✅ Best R²: {best_r2:.6f}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_methods():
    """Test tool methods"""
    print("\nTesting tool methods...")
    
    try:
        from semai.tools import HyperparameterTestingTool
        
        tool = HyperparameterTestingTool()
        
        # Test get_results_dataframe
        df = tool.get_results_dataframe()
        print(f"✅ get_results_dataframe(): {type(df).__name__}")
        
        # Test get_summary_report
        report = tool.get_summary_report()
        print(f"✅ get_summary_report(): {len(report)} chars")
        
        # Test get_best_config
        result = tool.get_best_config("NonExistent")
        print(f"✅ get_best_config(): returns {type(result)}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("HYPERPARAMETER TESTING TOOL - VERIFICATION TEST")
    print("="*70)
    
    results = {
        'Imports': test_imports(),
        'Performance Metrics': test_performance_metrics(),
        'Tool Initialization': test_tool_initialization(),
        'Quick Test Function': test_quick_hyperparameter_test(),
        'Tool Methods': test_tool_methods(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed! HyperparameterTesting tool is ready to use!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
