# Algorithm 1: Derivative Pricing Crew

**Start**

1. **Dataset Selection**
   - Option A: `dataset ← yahoo_finance_data(tickers, date_range)` → SPY, option chains, bid/ask quotes
   - Option B: `dataset ← alpha_vantage_data(symbols, interval)` → OHLCV data, technical indicators
   - Option C: `dataset ← csv_import(filepath)` → custom user data
   - Option D: `dataset ← market_data_api(source)` → real-time market feeds
   - Human selects: `dataset_choice ← select_from([A, B, C, D])`

2. `execute_crew()`
   - Level-1: data_extraction → data_generator → feature_engineering → model_constructor → meta_tuning → model_training
   - Level-2: performance_judge (assess) + documentation_writer (record)

3. **Human Decision Examples**
   
   **Example A: FEEDBACK**
   - Human: "Data quality issues detected. Re-run data_extraction with stricter validation."
   - `feedback_handler.process("stricter_validation_rules")`
   - goto 2 (re-execute with feedback)
   
   **Example B: FEEDBACK**
   - Human: "RMSE is too high. Try alternative features or hyperparameters."
   - `feedback_handler.process("test_polynomial_features + wider_hyperparameter_range")`
   - goto 2 (re-execute with modifications)
   
   **Example C: CONTINUE**
   - Human: "Results acceptable. Proceed to next iteration with expanded dataset."
   - `approval ← TRUE`
   - goto 2 (execute with larger dataset)
   
   **Example D: END**
   - Human: "Model performance meets production requirements. Finalize."
   - `l2_documentation_writer.finalize_report()`
   - `output ← ComputationalCrewDocumentation` (workflow, decisions, metrics, validation)
   - **End**
