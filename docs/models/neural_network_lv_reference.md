# Neural Network Local Volatility (NNLV) Model Reference

## Overview

The Neural Network Local Volatility model learns the local volatility surface directly from option price data using a neural network. Unlike traditional local volatility models that require Dupire's formula inversion, NNLV directly trains a neural network to map (S, t) to volatility.

## Mathematical Framework

### SDE Dynamics

The stock price follows:

$$dS_t = (r - d)S_t dt + \sigma(S_t, t; \theta)S_t dW_t$$

where:
- $r$ = risk-free interest rate
- $d$ = dividend yield
- $\sigma(S_t, t; \theta)$ = neural network-learned local volatility
- $\theta$ = network parameters
- $dW_t$ = standard Brownian motion increment

### Key Characteristics

1. **Network Input**: (S, t) - stock price and time
2. **Network Output**: $\sigma(S, t)$ - local volatility (positive)
3. **Activation**: softplus on output ensures $\sigma > 0$
4. **Training**: Direct MSE regression on call option prices
5. **Prediction**: Monte Carlo simulation with learned volatility

## Algorithm 1: Training (Direct Regression)

```
Input: Call option prices C, features X = [S, K, T, d, r, σ]
Output: Trained neural network f: (S, t) → σ

for epoch ← 1 to epochs do
    C_NN ← f(S, t)  // Network prediction of prices
    loss ← MSE(C, C_NN)
    loss.backward()
    optimizer.step()
end for
```

### Training Details

- **Input Features**: S (stock price) and T (time to maturity)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning_rate = 0.001)
- **Regularization**: Dropout (typically 20%)
- **Batch Size**: 32 (typical)
- **Epochs**: 50 (standard configuration)

## Algorithm 2: Simulation and Prediction

```
Input: Trained network f, features X = [S, K, T, d, r, σ]
       Initial stock price S₀, strike K, maturity T
       Number of paths L
Output: Option price C

for i ← 1 to L do
    S_0 ← S₀
    for t ← 0 to M-1 do
        C_t_NN ← f(S_t, t)  // Get volatility from network
        σ_t ← g(C_t_NN, t)  // Apply Dupire's formula (if needed)
        S_{t+1} ← S_t + (r - d)S_t Δt + σ_t S_t Δ√t Z_t
    end for
    
    // Calculate payoff at maturity
    Payoff_i ← max(S_T - K, 0)
end for

// Compute expected discounted payoff
C ← (1/L) Σ[i=1 to L] e^(-rT) * Payoff_i
```

### Simulation Details

- **Time Steps**: M = 252 (one trading year)
- **Time Increment**: Δt = T/M
- **Monte Carlo Paths**: L = 100 (standard), 200 (large)
- **Discretization**: Euler scheme
- **Brownian Increments**: ΔZ_t ~ N(0, Δt)

## Implementation Equations

### Euler Discretization for Path Simulation

$$S_{t+1} = S_t + (r - d)S_t \Delta t + \sigma(S_t, t; \theta)S_t \sqrt{\Delta t} \, Z_t$$

where $Z_t \sim \mathcal{N}(0, 1)$

### Option Price from Monte Carlo

$$C(S_0, K, T) = e^{-rT} \mathbb{E}[\max(S_T - K, 0)]$$

Estimated by:

$$\hat{C} = \frac{1}{L} \sum_{i=1}^{L} e^{-rT} \max(S_T^{(i)} - K, 0)$$

### Network Architecture

**Volatility Network** $f: (S, t) \rightarrow \sigma$

```
Input Layer: 2 neurons (S, T)
    ↓
Hidden Layer 1: 128 neurons, ReLU activation, 20% Dropout
    ↓
Hidden Layer 2: 64 neurons, ReLU activation, 20% Dropout
    ↓
Output Layer: 1 neuron, Softplus activation (ensures σ > 0)
```

## Hyperparameter Configurations

### Standard Configuration (nnlv_standard)
- Hidden layers: (128, 64)
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 50
- Batch size: 32
- Monte Carlo samples: 100
- Time steps: 252
- Activation: ReLU (hidden), Softplus (output)

### Large Configuration (nnlv_large)
- Hidden layers: (256, 128, 64)
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 100
- Batch size: 32
- Monte Carlo samples: 200
- Time steps: 252
- More capacity and precision

### Fast Configuration (nnlv_fast)
- Hidden layers: (64, 32)
- Dropout rate: 0.1 (10%)
- Learning rate: 0.001
- Epochs: 30
- Batch size: 64
- Monte Carlo samples: 50
- Time steps: 128
- Faster training and prediction

## Advantages

1. **Direct Data-Driven Learning**: No need for Dupire formula inversion
2. **Flexible Volatility Surface**: Can capture complex dependencies
3. **End-to-End Training**: Single network learns complete surface
4. **Monte Carlo Pricing**: Consistent with modern numerical methods
5. **Generalization**: Can interpolate to unseen (S, t) pairs

## Limitations

1. **Computational Cost**: Monte Carlo prediction requires many simulations
2. **Training Data**: Needs sufficient option prices across (S, t) space
3. **Stability**: Network must maintain positive volatility constraint
4. **Calibration**: Less transparent than analytical models

## Use Cases

1. **Volatility Surface Estimation**: Learn local vol from market data
2. **Exotic Option Pricing**: Flexible surface for complex derivatives
3. **Risk Management**: Dynamic volatility for scenario analysis
4. **Benchmark Comparison**: Compare against analytical methods

## Related Models

- **Black-Scholes**: Constant volatility benchmark
- **Neural Network SDE**: Parametric drift/diffusion networks
- **Deep Learning Net**: General neural network regression
- **Random Forest**: Non-parametric volatility estimation

## References

1. **Dupire's Formula**: Local volatility calibration from vanilla option prices
2. **Euler Discretization**: Standard scheme for SDE path simulation
3. **Monte Carlo**: Stochastic pricing methodology
4. **Neural Networks**: Universal function approximation for volatility surface

## Code Example

```python
from semai.builtin_models import get_model_registry

# Get registry and load model
registry = get_model_registry()
nnlv = registry.get_model("nnlv_standard")

# Train on data
nnlv.train(X_train, y_train, X_val, y_val)

# Predict prices
prices = nnlv.predict(X_test)

# Evaluate
metrics = nnlv.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"MAE: {metrics['mae']:.6f}")
print(f"R²: {metrics['r2']:.6f}")
```
