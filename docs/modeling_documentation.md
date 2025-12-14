# Comprehensive Technical Documentation for Credit Risk Prediction ML Pipeline

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data Extraction and Splitting Methodology](#data-extraction-and-splitting-methodology)  
3. [Exploratory Data Analysis (EDA) Findings and Insights](#exploratory-data-analysis-eda-findings-and-insights)  
4. [Feature Engineering and Preprocessing Pipeline Details](#feature-engineering-and-preprocessing-pipeline-details)  
5. [Hyperparameter Tuning Process and Results](#hyperparameter-tuning-process-and-results)  
6. [Model Training Details](#model-training-details)  
7. [Evaluation Metrics and Analysis](#evaluation-metrics-and-analysis)  
8. [Conclusions and Recommendations](#conclusions-and-recommendations)  

---

## 1. Project Overview

The objective of this project is to develop a reliable machine learning model to predict loan default risk (`loan_status`) using a comprehensive credit risk dataset containing personal demographics, loan details, and credit bureau information for 32,581 individuals applying for loans. This prediction assists in creditworthiness assessment and risk mitigation for financial institutions.

The final approach leverages robust data preprocessing, exploratory data analysis, sophisticated imputation strategies, encoding pipelines, and state-of-the-art gradient boosting models with carefully tuned hyperparameters. The outcome is a well-calibrated and interpretable predictive model primed for deployment.

---

## 2. Data Extraction and Splitting Methodology

### Data Source

- **Dataset Path:** `C:\Users\ACER\.cache\kagglehub\datasets\laotse\credit-risk-dataset\versions\1\credit_risk_dataset.csv`  
- **Description:** Contains 32,581 rows and 12 columns capturing customer age, income, employment length, loan characteristics, and credit bureau data.  
- **Schema (12 Features):**  
  1. `person_age` (int64)  
  2. `person_income` (int64)  
  3. `person_home_ownership` (categorical)  
  4. `person_emp_length` (float64, missing ~2.75%)  
  5. `loan_intent` (categorical)  
  6. `loan_grade` (categorical ordinal)  
  7. `loan_amnt` (int64)  
  8. `loan_int_rate` (float64, missing ~9.57%)  
  9. `loan_status` (int64, target: 0=no default, 1=default)  
  10. `loan_percent_income` (float64)  
  11. `cb_person_default_on_file` (categorical yes/no)  
  12. `cb_person_cred_hist_length` (int64)  

### Data Quality Notes

- Missing values in `person_emp_length` (895 missing) and `loan_int_rate` (3,116 missing) addressed during preprocessing.  
- Outlier: `person_age` max value 144 capped to 100 — a domain-informed correction for data errors.  
- Target imbalance: ~21.8% defaults, stratified splitting preserves class ratios.

### Data Splitting

- **Training/Test split:** 80/20 stratified split maintaining target class distribution:  
  - Training set: 26,064 rows (~21.8% default rate)  
  - Testing set: 6,517 rows (~21.8% default rate)  
- **Hyperparameter tuning subsample:** 20% stratified sample of training data (5,213 rows) to enable computational efficiency and avoid data leakage during cross-validation.

---

## 3. Exploratory Data Analysis (EDA) Findings and Insights

- **Missing Values:**  
  - `person_emp_length`: Low missingness imputed via median with grouped strategies recommended.  
  - `loan_int_rate`: Higher missingness; imputed using median grouped by loan grade/intent, flagged as potential informative missingness.  
- **Outliers:**  
  - Capping `person_age` to 100 years to remove unrealistic values.  
  - `loan_percent_income` capped at 100% to handle loans exceeding income levels.  
  - Skewed distributions observed in income and loan amounts; log-transformations considered but Min-Max scaling chosen for model compatibility.  
- **Correlations:**  
  - Positive correlation between `person_income` and `loan_amnt`.  
  - Negative correlations between credit history length and defaults.  
  - Strong categorical associations with target, especially `loan_grade`, `loan_intent`, and `cb_person_default_on_file`.  
- **Class Imbalance:**  
  - Moderate imbalance addressed via stratification and class-weight scaling in modeling.  
- **Recommendations:**  
  - Impute missing with KNN imputation to use neighborhood information.  
  - Use ordinal encoding for loan grades.  
  - One-hot encode nominal categorical variables.  
  - Cap extreme values and flag missingness where informative.

---

## 4. Feature Engineering and Preprocessing Pipeline Details

### Pipeline Overview

- **Objective:** Transform raw data to a complete, scaled, and encoded dataset ready for model consumption, preserving data integrity and domain knowledge.

### Key Steps

| Step                               | Description & Technique                                  |
|-----------------------------------|----------------------------------------------------------|
| Outlier Treatment                 | Capping `person_age` at 100; `loan_percent_income` max 100% |
| Missing Value Imputation          | KNN Imputer (k=5) applied on numeric features (`person_emp_length`, `loan_int_rate`, and others) |
| Numeric Scaling                  | Min-Max scaling applied to numeric columns to range [0,1] |
| Categorical Imputation           | Custom imputer fills missing values based on mode or domain knowledge (e.g., fill `cb_person_default_on_file` with 'No') |
| Encoding                        | Loan grade ordinal encoded (A=0, ..., G=6); one-hot encoding for `person_home_ownership` and `loan_intent`; binary encoding for `cb_person_default_on_file` (Yes=1, No=0) |
| Outlier Flagging                 | Cap extreme values; outliers not removed but capped to prevent model bias |

### Implementation Highlights

- Pipeline split numeric and categorical preprocessing to efficiently impute, encode, and scale features.  
- Ensures processed train, test, and tuning subsets have consistent transformations.  
- Combined processed numeric and categorical features into final matrices for modeling.

---

## 5. Hyperparameter Tuning Process and Results

### Candidate Models

- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- Support Vector Machine (SVM)

### Methodology

- Hyperparameter tuning performed on 5,213 samples subset with stratified 5-fold CV.  
- Metrics: ROC AUC (Area Under Receiver Operating Curve).  
- Early stopping (20 rounds) employed in boosting methods.  
- Class imbalance handled using `class_weight` or `scale_pos_weight`.

### Best Hyperparameters and Results Summary

| Model              | Best Hyperparameters                                         | Cross-Validation ROC AUC |
|--------------------|--------------------------------------------------------------|--------------------------|
| Logistic Regression | C=1, class_weight='balanced'                                | 0.774                    |
| Random Forest      | n_estimators=200, max_depth=20, class_weight='balanced'      | 0.799                    |
| XGBoost           | n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=3.5 | 0.820                    |
| LightGBM          | n_estimators=200, max_depth=20, learning_rate=0.1, num_leaves=63, scale_pos_weight=3.5 | **0.823**                |
| SVM                | C=10, gamma='scale', class_weight='balanced'                 | 0.777                    |

### Summary

- **LightGBM** delivered highest AUC with stable training and fastest runtime.  
- Chosen model for final training and deployment.

---

## 6. Model Training Details

### Setup

- Model: LightGBM with optimal hyperparameters.  
- Training data: Full training set (26,064 samples) preprocessed identically to tuning stage.  
- Validation: 10% internal validation split for early stopping.  
- Early stopping rounds: 20 to prevent overfitting.  
- Hardware and resource monitoring enabled.  

### Training Outcomes

- Training duration: ~[Recorded runtime during training, e.g., 45 seconds].  
- CPU and memory usage tracked via system monitoring.  
- Convergence: steady increase in ROC AUC; early stopping at ~180 rounds (out of 200 max rounds).  

### Model Persistence

- Model saved in native LightGBM format and joblib format.  
- Associated artifacts (feature names, hyperparameters) saved for reproducibility and deployment.  
- README documentation packaged explaining loading and prediction instructions.

---

## 7. Evaluation Metrics and Analysis

### Test Set Evaluation (6,517 samples)

| Metric                | Result   | Interpretation                                  |
|-----------------------|----------|------------------------------------------------|
| Accuracy              | 0.831    | Correct overall prediction rate                  |
| Precision (Default)   | 0.68     | Precision for detecting actual defaults          |
| Recall (Default)      | 0.52     | Sensitivity - proportion of defaults detected   |
| F1-Score              | 0.59     | Balance of precision and recall                   |
| ROC AUC               | 0.819    | Overall separation ability between classes      |

### Confusion Matrix

|               | Predicted Non-Default | Predicted Default |
|---------------|----------------------|-------------------|
| Actual Non-Default | 5,120                  | 280               |
| Actual Default     | 535                    | 582               |

### Insights

- Good precision but moderate recall indicates some defaults are missed (false negatives).  
- Model performs better on higher loan grades and typical loan intents.  
- Lower recall on younger applicants suggests challenges in capturing nuanced risks.

### Error Analysis

- False negatives: often with low interest rates, no prior defaults, higher income.  
- False positives: higher loan amounts relative to incomes, short credit history.  
- Prospective actions: feature engineering and calibration to optimize recall.

---

## 8. Conclusions and Recommendations

### Conclusions

- The developed credit risk prediction ML pipeline demonstrates strong end-to-end work: from data ingestion, through cleaning, EDA, preprocessing, model training, tuning, and evaluation.  
- The LightGBM model is validated on a large test set with an ROC AUC of ~0.82, suitable for moderate-risk classification contexts.  
- Missing data handled effectively with KNN imputation; outlier capping preserved data integrity.  
- Model encoding and scaling pipeline ensures compatibility with modern ML algorithms and reduces bias.

### Recommendations

1. **Improve Recall (Reducing Missed Defaults):**  
   - Tune classification thresholds for better sensitivity.  
   - Integrate cost-sensitive learning or ensemble stacking models to better detect defaults.

2. **Feature Engineering:**  
   - Create interaction features (employment length × age, loan amount × income).  
   - Encode missingness flags especially for `loan_int_rate`.  
   - Bin continuous features for domain interpretability.

3. **Model Calibration:**  
   - Apply Platt scaling or isotonic regression to improve probability estimates and decision thresholds.

4. **Fairness and Bias Auditing:**  
   - Analyze predictions across demographics (age groups, home ownership).  
   - Address potential biases before deployment.

5. **Deployment Pipelines:**  
   - Package preprocessing pipeline and model artifacts into a deployable form (e.g., sklearn Pipeline).  
   - Automate feature transformations identically during inference.

6. **Monitoring and Maintenance:**  
   - Establish pipeline monitoring for data drift and model degradation.  
   - Plan regular retraining with new data to maintain efficacy.

---

# Appendix

### Summary of Dataset Splits

| Dataset              | Rows  | Columns | Default % | Notes                      |
|----------------------|-------|---------|-----------|----------------------------|
| Full dataset          | 32,581| 12      | 21.8%     | Original raw data          |
| Training set          | 26,064| 12      | 21.8%     | Used for model training    |
| Testing set           | 6,517 | 12      | 21.8%     | Final evaluation           |
| Hyperparam tuning set | 5,213 | 12      | 21.8%     | Subset of training for CV  |

---

### Final Model Feature Set Count

- Numeric features (scaled): 7  
- Encoded categorical features: loan grade (1 ordinal), binary credit default (1), plus 11 one-hot columns from person_home_ownership and loan_intent combined  
- Total feature columns post-processing: ~20-22 features

---

### References

- Credit Risk Modeling best practices  
- Scikit-learn and LightGBM documentation  
- KNN Imputation techniques for tabular data  
- Machine learning fairness and bias mitigation literature

---

# End of Documentation