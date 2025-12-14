# Artificial Neuron Network (ANN) Model Reference

## Overview

The Artificial Neuron Network (ANN) model provides a direct neural network regression approach to European option pricing. Unlike complex stochastic models, ANN learns a direct mapping from option characteristics to prices through a multi-layer neural network with configurable depth.

## Mathematical Framework

### Core Formula (Single Hidden Layer)

$$c' = C/K = \varphi_2\left(a + \sum_{j} w_j \varphi_1\left(b_j + \sum_{i} \tilde{w}_{ij} x_i\right)\right)$$

where:
- $c'$ = normalized call price (C/K)
- $C$ = actual call option price
- $K$ = strike price
- $\varphi_1$ = activation function for hidden layer (typically ReLU)
- $\varphi_2$ = activation function for output layer (typically linear or sigmoid)
- $x_i$ = input variables (moneyness m, time to maturity τ, volatility σ)
- $\tilde{w}_{ij}$ = weights from input i to hidden neuron j
- $w_j$ = weights from hidden neuron j to output
- $a$ = output bias
- $b_j$ = bias for hidden neuron j

### Multi-Layer Extension

For networks with multiple hidden layers, functions are nested:

$$c' = \varphi_2(a + w^{(L)} \varphi_{L-1}(b^{(L-1)} + \ldots \varphi_1(b^{(1)} + \tilde{w}^{(1)} x) \ldots))$$

**Key Properties**:
- Nested functions preserve derivative sign through chain rule
- Proper activation function choice maintains monotonicity
- No change in inference for arbitrary number of layers

## Input Features

### Raw Features (X)

| Index | Feature | Symbol | Description |
|-------|---------|--------|-------------|
| 0 | Stock Price | S | Current spot price |
| 1 | Strike Price | K | Contract strike price |
| 2 | Time to Maturity | T | Expiration time (years) |
| 3 | Dividend Yield | d | Continuous dividend rate |
| 4 | Risk-Free Rate | r | Interest rate |
| 5 | Volatility | σ | Annualized volatility |

### Derived Features (Computed)

When `use_derived_features=True`:

| Derived | Formula | Interpretation |
|---------|---------|-----------------|
| Moneyness | m = S / K | Price relative to strike |
| Time to Maturity | τ = T | Same as input T |
| Volatility | σ | Same as input σ |

**Why Derived Features?**
- **Scale-invariant**: Moneyness handles different absolute prices
- **Dimensionless**: All inputs normalized for network stability
- **Economic meaning**: Standard inputs for option pricing
- **Reduced variance**: Normalized features improve training

## Network Architecture

### General Structure

```
Input Layer (3 neurons for derived features)
    ↓
Hidden Layer 1 (128 neurons, ReLU, 20% dropout)
    ↓
Hidden Layer 2 (64 neurons, ReLU, 20% dropout)
    ↓
Hidden Layer 3 (32 neurons, ReLU, 20% dropout)
    ↓
Output Layer (1 neuron, linear/sigmoid)
    ↓
Denormalization: output × K → price
```

### Activation Functions

| Layer | Function | Equation | Purpose |
|-------|----------|----------|---------|
| Hidden | ReLU | $\varphi(x) = \max(0, x)$ | Non-linearity, gradient flow |
| Output (Linear) | Identity | $\varphi(x) = x$ | Unbounded regression |
| Output (Sigmoid) | Sigmoid | $\varphi(x) = 1/(1+e^{-x})$ | [0,1] normalization |

### Dropout Regularization

- Applied after each hidden layer
- **Standard**: 20% dropout (drop 20% of neurons)
- **Deep**: 30% dropout (stronger regularization)
- **Purpose**: Prevent overfitting, improve generalization

## Normalization Strategy

### Output Normalization

**During Training:**
- Normalize targets: $y' = C / K$ (normalized price)
- Network learns normalized prices

**During Prediction:**
- Denormalize output: $C = c' \times K$ (recover prices)
- Ensures economically sensible predictions

**Benefits:**
1. **Scale invariance**: Works with any strike price
2. **Stability**: Network learns in [0, ∞) range
3. **Interpretability**: Network output is price-to-strike ratio
4. **Monotonicity**: Easier to maintain C ≥ 0

## Hyperparameter Configurations

### Standard Configuration (ann_standard)
- Hidden layers: (128, 64, 32)
- Activation: ReLU
- Output activation: Linear
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 100
- Batch size: 32
- Normalize output: True
- Use derived features: True
- **Total parameters**: ~24,000
- **Best for**: Balanced speed and accuracy

### Large Configuration (ann_large)
- Hidden layers: (256, 128, 64, 32)
- Activation: ReLU
- Output activation: Linear
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 150
- Batch size: 32
- **Total parameters**: ~95,000
- **Best for**: High accuracy on large datasets

### Deep Configuration (ann_deep)
- Hidden layers: (256, 256, 128, 64, 32)
- Activation: ReLU
- Output activation: Linear
- Dropout rate: 0.3 (30%)
- Learning rate: 0.0005
- Epochs: 200
- Batch size: 32
- **Total parameters**: ~180,000
- **Best for**: Maximum model capacity, complex relationships

### Sigmoid Configuration (ann_sigmoid)
- Hidden layers: (128, 64, 32)
- Activation: ReLU
- Output activation: Sigmoid
- Dropout rate: 0.2 (20%)
- **Total parameters**: ~24,000
- **Best for**: Constrained outputs in [0, 1]

### Simple Configuration (ann_simple)
- Hidden layers: (64, 32)
- Activation: ReLU
- Output activation: Linear
- Dropout rate: 0.15 (15%)
- Learning rate: 0.001
- Epochs: 50
- Batch size: 64
- **Total parameters**: ~5,000
- **Best for**: Quick training, real-time constraints

## Training Algorithm

### Data Preparation

```
Input: Raw features X = [S, K, T, d, r, σ], prices C

1. Derive features:
   x = [m, τ, σ] where m = S/K, τ = T
   
2. Normalize output:
   C' = C / K
   
3. Split into batches
```

### Training Loop

```
for epoch = 1 to epochs:
    for batch in training_data:
        // Forward pass
        ŷ = network(batch_X_derived)
        
        // Compute loss (MSE)
        loss = mean((C' - ŷ)²)
        
        // Backpropagation
        ∇_θ = gradient(loss, θ)
        
        // Update parameters
        θ ← θ - α·∇_θ  (Adam optimizer)
    
    // Validation
    val_loss = evaluate(X_val, C'_val)
    
    if verbose:
        print(f"Epoch {epoch}: loss={loss:.6f}, val_loss={val_loss:.6f}")
```

### Loss Function

Mean Squared Error on normalized prices:

$$\text{Loss} = \frac{1}{n}\sum_{i=1}^{n} \left(C'_i - \hat{c}'_i\right)^2$$

where $C'_i = C_i / K_i$ (normalized market prices) and $\hat{c}'_i$ (network predictions).

## Prediction Process

### Forward Pass

```
Input: Features X = [S, K, T, d, r, σ]

1. Derive features:
   x = [S/K, T, σ]
   
2. Network inference:
   c' = network(x)
   
3. Denormalize:
   C = c' × K
   
4. Clamp to [0, ∞):
   C = max(C, 0)
   
Output: Predicted price C
```

## Advantages

1. **Simple Architecture**: Easy to understand and implement
2. **Fast Training**: No Monte Carlo simulation needed
3. **Direct Regression**: Learns price function directly
4. **Scalable**: Works with large datasets
5. **Flexible**: Multiple hidden layers capture complex patterns
6. **Normalized**: Output normalization ensures stability
7. **Interpretable**: Clear input-output relationship

## Limitations

1. **No Theoretical Consistency**: Doesn't enforce option pricing constraints
2. **Extrapolation Risk**: May behave poorly outside training domain
3. **No Arbitrage Guarantee**: Can violate no-arbitrage conditions
4. **Overfitting**: Requires careful regularization
5. **Black Box**: Limited interpretability compared to analytical models
6. **Data Dependency**: Quality depends on training data
7. **Static Model**: Doesn't capture time-evolution explicitly

## Computational Complexity

### Training Complexity

- **Forward pass per sample**: O(L × H²) where L = layers, H = hidden units
- **Backward pass**: Same as forward
- **Per epoch**: O(n × L × H²) where n = number of samples
- **Full training**: O(epochs × n × L × H²)

### Prediction Complexity

- **Per sample**: O(L × H²)
- **Batch of m samples**: O(m × L × H²) (vectorized)
- **Typical time**: ~1-10ms per sample on CPU, <1ms on GPU

### Memory Usage

- **Model parameters**: Sum of (input_i × hidden_i + bias) for each layer
- **Standard config**: ~24,000 parameters ≈ 96 KB
- **Large config**: ~95,000 parameters ≈ 380 KB
- **Training batch**: ~8-64 samples × features

## Comparison with Other Models

| Model | Type | Training | Speed | Accuracy | Constraints |
|-------|------|----------|-------|----------|-------------|
| **Black-Scholes** | Analytical | None | ★★★★★ | ★★ | Many |
| **Deep Learning Net** | NN Direct | Fast | ★★★★ | ★★★ | None |
| **ANN** | NN Direct | Fast | ★★★★ | ★★★ | None |
| **NNLV** | NN + Vol | Slow | ★★★ | ★★★★ | None |
| **SDENN** | NN + SDE | Very Slow | ★ | ★★★★★ | None |
| **2D-NN** | NN + SDE2D | Very Slow | ★ | ★★★★★ | None |

## Feature Engineering Insights

### Why Moneyness Works

- **Option value scale**: Intrinsic + extrinsic components
- **At-the-money**: m = 1, option has full time value
- **In-the-money**: m > 1, option has intrinsic value
- **Out-of-money**: m < 1, option is pure time value

### Why Normalized Prices Work

- **Dimensionless**: Price/Strike is scale-invariant
- **Bounded range**: c'/K ∈ [0, ∞) with practical bounds
- **Economic meaning**: Ratio of value to contract size
- **Network stability**: Easier for sigmoid/tanh to learn

## Use Cases

1. **Real-Time Pricing**: Fast inference for trading systems
2. **Batch Valuation**: Price large option portfolios
3. **Greeks Approximation**: Compute delta/gamma via finite diff
4. **Calibration**: Find implied volatility
5. **Market Making**: Quick price updates
6. **Risk Management**: Portfolio rebalancing
7. **Training Data**: Generate synthetic prices for other models

## Advanced Topics

### Computing Greeks (Numerical Differentiation)

**Delta** (dC/dS):
```python
eps = 0.01
S_up = S + eps
S_down = S - eps
delta = (model.predict(S_up) - model.predict(S_down)) / (2 * eps)
```

**Gamma** (d²C/dS²):
```python
gamma = (model.predict(S_up) - 2*model.predict(S) + model.predict(S_down)) / (eps ** 2)
```

**Vega** (dC/dσ):
```python
eps = 0.001
sigma_up = sigma + eps
sigma_down = sigma - eps
vega = (model.predict(sigma_up) - model.predict(sigma_down)) / (2 * eps)
```

### Ensemble with Other Models

```python
# Combine ANN with Black-Scholes
weights = [0.7, 0.3]
price_ann = ann_model.predict(X)
price_bs = bs_model.predict(X)
ensemble_price = weights[0] * price_ann + weights[1] * price_bs
```

### Domain Adaptation

Train on one underlying asset, fine-tune on another:

```python
# Pre-train on liquid stock options
model_pretrained = ann_large

# Fine-tune on less liquid options
model_finetuned = clone(model_pretrained)
model_finetuned.train(X_illiquid, y_illiquid, epochs=10)
```

## Model Selection Guide

Choose **ANN** when:
- ✓ Need real-time predictions
- ✓ Have large, clean datasets
- ✓ Want simple, interpretable model
- ✓ Computational resources are limited
- ✓ Don't need theoretical guarantees

Choose **SDENN** or **2D-NN** when:
- ✓ Need maximum accuracy
- ✓ Have sufficient computational resources
- ✓ Need volatility dynamics
- ✓ Working with exotic options

Choose **Black-Scholes** when:
- ✓ Need closed-form solutions
- ✓ Have European vanilla options
- ✓ Constant volatility assumption holds
- ✓ Need theoretical guarantees

## Code Example

```python
from semai.builtin_models import get_model_registry

# Load model
registry = get_model_registry()
ann = registry.get_model("ann_standard")

# Train
ann.train(X_train, y_train, X_val, y_val, verbose=True)

# Get model info
info = ann.get_model_info()
print(f"Total parameters: {info['total_parameters']}")
print(f"Hidden layers: {info['hidden_layers']}")
print(f"Final loss: {info['training_history']['final_loss']:.6f}")

# Predict
prices = ann.predict(X_test)

# Evaluate
metrics = ann.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.6f}")
print(f"Test MAE: {metrics['mae']:.6f}")

# Compute delta via finite differences
eps = 0.01
X_up = X_test.copy()
X_up[:, 0] += eps  # Bump stock price
prices_up = ann.predict(X_up)
delta = (prices_up - prices) / eps
print(f"Average Delta: {delta.mean():.4f}")
```

## References

1. **McCulloch & Pitts (1943)**: Foundational neural network theory
2. **Backpropagation (1986)**: Training algorithm for deep networks
3. **Option Pricing with Neural Networks**: Direct regression approaches
4. **ReLU Activation (2011)**: Improved training for deep networks
5. **Batch Normalization (2015)**: Stable training for deep models

## Performance Notes

### Training Time (Standard Config, 240k samples)
- **CPU**: 5-15 minutes per 100 epochs
- **GPU (CUDA)**: 1-3 minutes per 100 epochs
- **Memory**: ~2-4 GB RAM

### Prediction Time
- **Single sample**: 0.1-1 ms (CPU), <0.1 ms (GPU)
- **1000 samples**: 100-1000 ms (CPU), 10-100 ms (GPU)
- **10000 samples**: 1-10 seconds (CPU), 0.1-1 seconds (GPU)

### Accuracy Range
- **RMSE**: 0.1-1.0 (depends on price scale)
- **R²**: 0.90-0.99 (depends on data quality)
- **MAE**: 0.05-0.5 (similar to RMSE)
