# SDENN (SDE Neural Network) Model Reference

## Overview

The SDE Neural Network (SDENN) model represents an advanced approach to derivative pricing that optimizes the entire stochastic differential equation end-to-end using stochastic gradient descent. Unlike NNLV which performs offline volatility calibration, SDENN integrates Dupire's formula directly into the optimization process, enabling simultaneous optimization of price predictions and SDE simulation.

## Key Innovation: End-to-End SDE Optimization

**NNLV Approach** (Offline):
1. Train network on prices: C(S, K, T; θ)
2. Extract volatility using Dupire's formula: σ(C)
3. Simulate SDE with extracted volatility
4. Sub-optimal: Network and simulation optimized separately

**SDENN Approach** (End-to-End):
1. Train network on prices: C(S, K, T; θ)
2. Compute volatility via Dupire's formula: σ(C)
3. Simulate SDE with learned volatility
4. Compute Monte Carlo payoffs
5. Optimize network so that SDE simulation matches market prices
6. Optimal: Joint optimization of network and simulation

## Mathematical Framework

### SDE with Dupire-Derived Volatility

Stock price dynamics:

$$dS_t = (r - d)S_t dt + \sigma(S_t, t; \theta)S_t dW_t$$

where volatility comes from Dupire's formula applied to the network output:

$$\sigma^2(S, t; \theta) = \frac{2\frac{\partial C}{\partial T} + (r-d)K\frac{\partial C}{\partial K} + dC K^{-2}\frac{\partial^2 C}{\partial K^2}}{K^2 \frac{\partial^2 C}{\partial K^2}}$$

### Dupire's Formula Components

**Numerator** (theta + rho + gamma terms):
- $2\frac{\partial C}{\partial T}$: Theta (time decay)
- $(r-d)K\frac{\partial C}{\partial K}$: Rho-like term (drift adjustment)
- $dC K^{-2}\frac{\partial^2 C}{\partial K^2}$: Convexity term

**Denominator** (gamma term):
- $K^2 \frac{\partial^2 C}{\partial K^2}$: Gamma (curvature)

### Network Architecture

**Price Network** $f: (S, K, T) \rightarrow C$

```
Input Layer: 3 neurons (S, K, T)
    ↓
Hidden Layer 1: 128 neurons, ReLU activation, 20% Dropout
    ↓
Hidden Layer 2: 64 neurons, ReLU activation, 20% Dropout
    ↓
Output Layer: 1 neuron, Softplus activation (ensures C > 0)
```

The output of this network is then differentiated (numerically) to extract volatility via Dupire's formula.

## Algorithm: End-to-End SDE Optimization

### Training Procedure

```
Input: Call option prices {C_i}, features X_i = [S_i, K_i, T_i, d, r, σ]
Initialize: Neural network θ with random weights

for epoch ← 1 to epochs do
    for batch in training_data do
        // Forward: Get network prices
        C_NN = f(S, K, T; θ)
        
        // Compute Dupire volatility from network output
        σ(t) = DUPIRE(C_NN, ∂C_NN/∂T, ∂C_NN/∂K, ∂²C_NN/∂K²)
        
        // Simulate SDE paths with learned volatility
        for j ← 1 to L do
            S_T^(j) ← SIMULATE_EULER(S_0, σ(t), T)
            Payoff^(j) ← max(S_T^(j) - K, 0)
        end for
        
        // Compute Monte Carlo price
        C_SDE = e^(-rT) * (1/L) * Σ Payoff^(j)
        
        // Loss: Match network output to SDE simulation
        loss = MSE(C_NN, C_SDE)
        
        // Backpropagate through entire computation graph
        θ ← θ - α * ∇_θ loss
    end for
end for
```

### Key Differences from NNLV

| Aspect | NNLV | SDENN |
|--------|------|-------|
| **Training Target** | Direct prices C | SDE-simulated prices |
| **Volatility Role** | Extracted post-training | Part of training loss |
| **Optimization Scope** | Price network only | Price network + SDE path |
| **Gradient Flow** | Stops at network output | Flows through simulation |
| **Loss Computation** | Price regression | SDE payoff matching |
| **Calibration** | Offline Dupire | Online through training |

## Implementation Details

### Numerical Differentiation for Dupire's Formula

Compute partial derivatives using finite differences:

$$\frac{\partial C}{\partial T} \approx \frac{C(S, K, T + \epsilon) - C(S, K, T - \epsilon)}{2\epsilon}$$

$$\frac{\partial C}{\partial K} \approx \frac{C(S, K + \epsilon) - C(S, K - \epsilon)}{2\epsilon}$$

$$\frac{\partial^2 C}{\partial K^2} \approx \frac{C(S, K + \epsilon) - 2C(S, K) + C(S, K - \epsilon)}{\epsilon^2}$$

where $\epsilon = 10^{-3}$ (configurable).

### Euler Discretization for Path Simulation

$$S_{t+1} = S_t + (r - d)S_t \Delta t + \sigma(S_t, t; \theta)S_t \sqrt{\Delta t} \, Z_t$$

where:
- $\Delta t = T / M$ (time step)
- $M = 252$ (time steps, one trading year)
- $Z_t \sim \mathcal{N}(0, 1)$ (standard normal)

### Option Price from Monte Carlo

$$C(S_0, K, T) \approx e^{-rT} \frac{1}{L} \sum_{i=1}^{L} \max(S_T^{(i)} - K, 0)$$

where $L = 100$ (standard), $200$ (large) is the number of paths.

## Hyperparameter Configurations

### Standard Configuration (sdenn_standard)
- Price network layers: (128, 64)
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 100
- Batch size: 32
- Monte Carlo samples: 100
- Time steps: 252
- Use Dupire: True
- **Best for**: Balanced performance and computational cost

### Large Configuration (sdenn_large)
- Price network layers: (256, 128, 64)
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 150 (more training for larger network)
- Batch size: 32
- Monte Carlo samples: 200
- Time steps: 252
- **Best for**: High accuracy requirements, larger datasets

### Fast Configuration (sdenn_fast)
- Price network layers: (64, 32)
- Dropout rate: 0.15 (15%)
- Learning rate: 0.001
- Epochs: 50
- Batch size: 64
- Monte Carlo samples: 50
- Time steps: 128
- **Best for**: Quick experimentation, real-time constraints

## Advantages

1. **End-to-End Optimization**: Joint optimization through entire SDE pipeline
2. **Implicit Volatility Learning**: Automatically captures volatility surface from prices
3. **Consistency**: Network and SDE optimized for same objective
4. **Flexibility**: Can handle complex market dynamics
5. **Dupire Integration**: Leverage classical local volatility framework
6. **Better Generalization**: Joint optimization often improves robustness

## Limitations

1. **Computational Cost**: Most expensive among all models (SDE simulation + backprop)
2. **Numerical Stability**: Dupire formula can be sensitive to numerical differentiation
3. **Training Time**: Requires 100-150 epochs with Monte Carlo at each step
4. **Hyperparameter Tuning**: More parameters to tune than simpler models
5. **Memory Usage**: Stores paths for all samples in batch during training

## Computational Complexity

Per epoch:
- **Batches**: ceil(n_samples / batch_size)
- **Per batch**: SDE simulation for L paths × time_steps = O(L × M) operations
- **Per epoch**: O(n_samples × L × M) operations
- **Full training**: O(epochs × n_samples × L × M)

For standard config with 240,000 samples:
- ~58 million operations per epoch
- ~5.8 billion operations for 100 epochs
- **Typical training time**: 30-60 minutes (GPU), 2-4 hours (CPU)

## Use Cases

1. **Volatility Surface Learning**: Capture complex smile/skew effects
2. **Exotic Pricing**: Flexible framework for path-dependent options
3. **Market Calibration**: Learn volatility from market prices end-to-end
4. **Scenario Analysis**: Simulate paths with learned volatility
5. **Sensitivity Analysis**: Understand price sensitivity to spot/time

## Comparison with Other Models

| Model | Training | Simulation | Speed | Accuracy |
|-------|----------|-----------|-------|----------|
| **Black-Scholes** | None | Analytical | ★★★★★ | ★★ |
| **Deep Learning Net** | Direct | N/A | ★★★★ | ★★★ |
| **Neural Network SDE** | SDE paths | Monte Carlo | ★★ | ★★★ |
| **NNLV** | Offline Dupire | SDE simulation | ★★★ | ★★★★ |
| **SDENN** | End-to-end SDE | SDE simulation | ★ | ★★★★★ |

## References

1. **Dupire, B.** (1994). "Pricing with a smile." *Risk Magazine*.
2. **Local Volatility**: Classic framework for capturing smile effects
3. **Neural Network Integration**: End-to-end optimization via backpropagation
4. **Euler Discretization**: Standard approach for SDE simulation
5. **Monte Carlo Methods**: Stochastic pricing for complex derivatives

## Code Example

```python
from semai.builtin_models import get_model_registry

# Get registry and load model
registry = get_model_registry()
sdenn = registry.get_model("sdenn_standard")

# Train on data
print("Training SDENN (this may take a while)...")
sdenn.train(X_train, y_train, X_val, y_val, verbose=True)

# Predict prices
prices = sdenn.predict(X_test)

# Evaluate
metrics = sdenn.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"MAE: {metrics['mae']:.6f}")
print(f"R²: {metrics['r2']:.6f}")

# Compare with other models
model_ids = ["blackscholes_benchmark", "dl_net_optimized", "nnlv_standard", "sdenn_standard"]
comparison = registry.compare_models(model_ids, X_test, y_test)
print(comparison)
```

## Advanced Topics

### Volatility Stability

The Dupire formula can produce unstable volatility estimates. SDENN addresses this by:
- Bounding volatility: $\sigma \in [0.01, 2.0]$ (1% - 200%)
- Using small epsilon for differentiation: $\epsilon = 10^{-3}$
- Adding dropout for regularization
- Training with multiple Monte Carlo samples

### Gradient Flow Through Simulation

The backpropagation path:
1. MSE loss between network output and SDE price
2. Gradient flows back through Monte Carlo averaging
3. Gradient flows through Euler discretization
4. Gradient flows through Dupire formula (numerical differentiation)
5. Updates network parameters

This complex gradient flow is what makes SDENN powerful but computationally expensive.

### Calibration vs. Prediction

- **Calibration**: Training on market option prices
- **Prediction**: Using trained network for pricing new options
- **Consistency**: SDE simulation ensures consistent pricing methodology
