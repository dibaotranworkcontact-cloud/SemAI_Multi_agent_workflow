# Derivative Pricing Dataset Summary

## Dataset Overview
Generated synthetic call option prices using the Black-Scholes model with realistic market parameters.

### Dataset Specifications
- **Total Samples**: 300,000
- **Training Set**: 240,000 samples (80%)
- **Validation Set**: 60,000 samples (20%)
- **Features**: 7 (6 input parameters + 1 target variable)

### Parameter Ranges (Table 1)

| Parameter | Range | Description |
|-----------|-------|-------------|
| **S** (Stock Price) | $10 – $500 | Current underlying stock price |
| **K** (Strike Price) | $7 – $650 | Option exercise price |
| **T** (Maturity) | 1 day – 3 years | Time to expiration (converted to years) |
| **q** (Dividend Rate) | 0% – 3% | Annual dividend yield |
| **r** (Risk-free Rate) | 1% – 3% | Annual risk-free interest rate |
| **σ** (Volatility) | 5% – 90% | Annualized stock volatility |
| **C** (Call Price) | $0 – $328 | Target: European call option price |

### Data Files

#### Training Dataset
- **File**: `derivative_pricing_train.csv`
- **Size**: ~31.24 MB
- **Samples**: 240,000
- **Location**: `src/semai/derivative_pricing_train.csv`

#### Validation Dataset
- **File**: `derivative_pricing_test.csv`
- **Size**: ~7.81 MB
- **Samples**: 60,000
- **Location**: `src/semai/derivative_pricing_test.csv`

### Data Format
Each CSV file contains 7 columns:
```
S,K,T,q,r,sigma,C
222.51,464.69,2.89,0.026,0.012,0.581,34.49
392.49,190.40,0.569,0.030,0.017,0.343,197.39
...
```

### Methodology

#### Black-Scholes Formula
The call option prices are calculated using the Black-Scholes formula:

```
C = S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)

where:
  d1 = [ln(S/K) + (r-q+0.5*σ²)*T] / (σ*√T)
  d2 = d1 - σ*√T
  N(x) = cumulative standard normal distribution
```

#### Realistic Constraints
- Strike prices are chosen in the vicinity of stock prices for realistic scenarios
- Parameter values are randomly sampled from uniform distributions
- All calculations follow standard financial market conventions

### Key Features
✅ **Realistic Parameters**: All ranges based on actual market conditions
✅ **Train/Validation Split**: 80/20 split for proper model evaluation
✅ **Standardized Format**: CSV format for easy integration with ML frameworks
✅ **Large Scale**: 300,000 samples sufficient for deep learning models
✅ **Reproducible**: Generated with fixed random seed for reproducibility

### Usage

#### Loading the Dataset
```python
import pandas as pd

# Load training data
train_data = pd.read_csv('derivative_pricing_train.csv')

# Load validation data
validation_data = pd.read_csv('derivative_pricing_test.csv')

# Split features and target
X_train = train_data[['S', 'K', 'T', 'q', 'r', 'sigma']]
y_train = train_data['C']

X_val = validation_data[['S', 'K', 'T', 'q', 'r', 'sigma']]
y_val = validation_data['C']
```

#### Expected Model Task
- **Input Features**: 6 (S, K, T, q, r, sigma)
- **Target Variable**: 1 (C - Call option price)
- **Task Type**: Regression
- **Typical Range**: Model should predict prices between $0 and $328

### Next Steps
This dataset is ready for:
1. **Model Training**: Train pricing prediction models
2. **Feature Analysis**: Analyze feature importance and correlations
3. **Neural Network Development**: Build deep learning models for derivative pricing
4. **Validation Testing**: Evaluate model performance on held-out validation set
5. **Risk Analysis**: Study model errors and volatility predictions

---
**Generated**: December 9, 2025
**Method**: Black-Scholes European Call Option Pricing
**Total File Size**: ~39 MB (compressed for efficient storage)
