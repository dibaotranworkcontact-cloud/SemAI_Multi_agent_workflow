# 2D Neural Network (2D-NN) SDE Model Reference

## Overview

The 2D Neural Network (2D-NN) model represents a sophisticated approach to option pricing through a coupled system of neural network-driven SDEs. Unlike previous models that model either price or volatility separately, 2D-NN simultaneously models both through two interconnected stochastic differential equations with correlated Brownian motions.

## Mathematical Framework

### Coupled SDE System

**Stock Price Dynamics:**
$$dS_t = f_1(S_t, Y_t, t; \theta) dt + f_2(S_t, Y_t, t; \theta) dW_t^S$$

**Stochastic Volatility Dynamics:**
$$dY_t = f_3(S_t, Y_t, t; \theta) dt + f_4(S_t, Y_t, t; \theta) dW_t^Y$$

where:
- $f: (s, y, t; \theta) \rightarrow \mathbb{R}^4$ is a neural network with 4 outputs
- $f_1, f_2, f_3, f_4$ are the four neural network outputs
- $W_t^S, W_t^Y$ are correlated Brownian motions with correlation $\rho$
- $\theta$ are network parameters
- $\rho$ (correlation) and $Y_0$ (initial volatility) are trained

### Component Functions

| Component | Role | Typical Range | Constraints |
|-----------|------|----------------|-------------|
| $f_1(S, Y, t)$ | Drift of S (expected return) | ~$(r-d)S$ | Linear in S |
| $f_2(S, Y, t)$ | Diffusion of S (volatility) | ~$\sqrt{Y} S$ | Must be positive |
| $f_3(S, Y, t)$ | Drift of Y (mean reversion) | ~$\kappa(\bar{Y} - Y)$ | Can be negative |
| $f_4(S, Y, t)$ | Diffusion of Y (vol of vol) | ~$\sigma_Y\sqrt{Y}$ | Must be positive |

### Correlated Brownian Motions

Generate correlated Brownian increments:

$$dW_t^Y = \rho \, dW_t^S + \sqrt{1 - \rho^2} \, dW_t^{\perp}$$

where:
- $\rho \in [-1, 1]$ is the correlation (trained parameter)
- $dW_t^{\perp}$ is independent standard Brownian motion
- Negative $\rho$ captures "leverage effect" (price down → volatility up)

## Trained Parameters

### Correlation Parameter $\rho$

The correlation between the two Brownian motions:
- **Range**: $[-0.99, 0.99]$ (clipped for stability)
- **Interpretation**: How price and volatility changes are related
- **Typical Value**: -0.5 (moderate negative correlation, leverage effect)
- **Learns**: How much volatility tends to spike when price drops

### Initial Volatility $Y_0$

The initial level of the volatility state variable:
- **Range**: $(0, \infty)$ (constrained positive)
- **Interpretation**: Starting point for mean-reverting volatility
- **Typical Value**: 0.04 (2% volatility, squared)
- **Learns**: Optimal starting point for volatility state

## Network Architecture

### Multi-Output Neural Network

**Inputs**: $(S, Y, t)$ - stock price, volatility state, time

**Architecture**:
```
Input Layer: 3 neurons (S, Y, t)
    ↓
Hidden Layer 1: 128 neurons, ReLU activation, 20% Dropout
    ↓
Hidden Layer 2: 64 neurons, ReLU activation, 20% Dropout
    ↓
Output Layer: 4 neurons
    ├─ f₁: Linear activation (drift, can be negative)
    ├─ f₂: Softplus activation (diffusion S, must be positive)
    ├─ f₃: Linear activation (drift Y, can be negative)
    └─ f₄: Softplus activation (diffusion Y, must be positive)
```

### Output Activation Functions

| Output | Activation | Reason |
|--------|-----------|--------|
| $f_1$ | Linear | Drift can be positive or negative |
| $f_2$ | Softplus | Diffusion must be positive |
| $f_3$ | Linear | Mean-reversion can push either direction |
| $f_4$ | Softplus | Volatility of volatility must be positive |

## Algorithm: Coupled SDE Simulation and Training

### Simulation Procedure (Euler Discretization)

```
Input: S₀, K, T, Y₀, ρ, network parameters θ
Initialize: S₀, Y₀

for path i ← 1 to L do
    for time step t ← 0 to M-1 do
        // Get network outputs for current state
        [f₁, f₂, f₃, f₄] ← f(Sₜ, Yₜ, t; θ)
        
        // Generate correlated Brownian increments
        ΔW_S ~ N(0, Δt)
        ΔW_Y = ρ·ΔW_S + √(1-ρ²)·ΔW_⊥
        
        // Update stock price
        ΔS = f₁·Δt + f₂·√Δt·ΔW_S
        Sₜ₊₁ = Sₜ + ΔS
        
        // Update volatility state (ensure positive)
        ΔY = f₃·Δt + f₄·√Δt·ΔW_Y
        Yₜ₊₁ = max(Yₜ + ΔY, 0.001)
    end for
    
    // Compute payoff at maturity
    Payoff_i = max(S_T - K, 0)
end for

// Monte Carlo price estimate
C = e^(-rT) · (1/L) · Σ Payoff_i
```

### Training Procedure

```
Input: Training data {(Sᵢ, Kᵢ, Tᵢ, Cᵢ)}, validation data
Initialize: Network weights θ, ρ ~ 0, Y₀ ~ 0.04

for epoch ← 1 to epochs do
    for batch in training_data do
        // For each sample in batch:
        for each training sample do
            // Simulate coupled paths with current parameters
            S_paths, Y_paths ← SIMULATE_2D(S₀, K, T, ρ, Y₀; θ)
            
            // Compute Monte Carlo price
            C_SDE = e^(-rT) · mean(max(S_T - K, 0))
        end for
        
        // Loss: MSE between market and SDE prices
        loss = MSE(C_market, C_SDE)
        
        // Backpropagate through network
        θ ← θ - α · ∇_θ loss
    end for
    
    // Validation step (fewer paths for speed)
    for validation sample do
        Compute validation loss
    end for
end for
```

## Hyperparameter Configurations

### Standard Configuration (2d_nn_standard)
- Network layers: (128, 64)
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 100
- Batch size: 32
- Monte Carlo samples: 100
- Time steps: 252
- Y₀ initial: 0.04 (2% vol)
- ρ initial: 0.0
- **Best for**: Balanced model with stable training

### Large Configuration (2d_nn_large)
- Network layers: (256, 128, 64)
- Dropout rate: 0.2 (20%)
- Learning rate: 0.001
- Epochs: 150 (more training for larger network)
- Batch size: 32
- Monte Carlo samples: 200
- Time steps: 252
- Y₀ initial: 0.04
- ρ initial: -0.5 (leverage effect)
- **Best for**: High accuracy, larger datasets, capture complex dynamics

### Fast Configuration (2d_nn_fast)
- Network layers: (64, 32)
- Dropout rate: 0.15 (15%)
- Learning rate: 0.001
- Epochs: 50
- Batch size: 64
- Monte Carlo samples: 50
- Time steps: 128
- Y₀ initial: 0.04
- ρ initial: 0.0
- **Best for**: Quick experimentation, real-time constraints

## Key Advantages

1. **Stochastic Volatility**: Captures volatility clustering and smile effects
2. **Leverage Effect**: Correlation parameter models price-volatility relationship
3. **Fully Parameterized**: All dynamics learned from data via neural networks
4. **Mean Reversion**: $f_3$ can naturally learn mean-reverting behavior
5. **Non-Linear Effects**: Network captures complex dependencies between S and Y
6. **Coupled Learning**: Price and volatility optimized jointly
7. **Flexible Dynamics**: Can approximate various classical models (Heston, CEV, etc.)

## Computational Complexity

### Per Epoch
- **Forward pass**: Simulate L paths × M time steps × 4 network outputs
- **Operations per sample**: O(L × M)
- **Full epoch**: O(n_samples × L × M)

### Typical Values (Standard Config, 240k samples)
- 240,000 samples × 100 paths × 252 steps = 6.048 billion operations
- **Training time**: 1-2 hours (GPU), 6-12 hours (CPU)
- **Prediction time**: ~2-5 seconds per sample (requires 100 MC simulations)

## Comparison: 2D-NN vs Other Stochastic Models

| Aspect | SDENN | NNLV | 2D-NN |
|--------|-------|------|-------|
| **Volatility** | Dupire extracted | Network learned | SDE state |
| **Time Dependency** | Implicit via Dupire | Network input | Explicit |
| **Correlation** | N/A | N/A | Trained ρ |
| **State Variables** | 1 (S) | 1 (S) | 2 (S, Y) |
| **Complexity** | Medium | Medium | High |
| **Flexibility** | Medium | Medium | High |
| **Accuracy** | High | High | Very High |
| **Speed** | Medium | Slow | Slow |

## Theoretical Interpretation

### Heston-Like Model
The 2D-NN can learn dynamics similar to Heston's stochastic volatility:
- $f_1(S, Y, t) \approx (r-d)S$ (lognormal drift)
- $f_2(S, Y, t) \approx \sqrt{Y}S$ (volatility proportional to $\sqrt{Y}$)
- $f_3(S, Y, t) \approx \kappa(\bar{Y} - Y)$ (mean reversion)
- $f_4(S, Y, t) \approx \xi\sqrt{Y}$ (volatility of volatility)

### CEV-Like Model
Or it could learn constant elasticity of variance effects:
- $f_2(S, Y, t) \approx Y \cdot S^{\beta}$ (power law volatility)

### Jump Diffusion Effects
The network could even approximate jump-like behavior through sharp non-linearities.

## Practical Considerations

### Initialization
- **Y₀**: Initialize to 0.04 (2% annual volatility squared)
- **ρ**: Initialize to 0.0 or -0.5 (leverage effect)
- **Network weights**: Standard random normal initialization

### Stability Measures
1. **Positive constraints**: Y ≥ 0.001 (keep volatility state positive)
2. **Diffusion bounds**: Softplus ensures f₂, f₄ > 0
3. **Correlation clipping**: ρ ∈ [-0.99, 0.99]
4. **Dropout regularization**: 20% dropout prevents overfitting

### Numerical Precision
- **Time discretization**: 252 steps (business day granularity)
- **Small variance threshold**: 0.001 prevents numerical issues
- **Batch normalization**: Consider adding for deeper networks

## Use Cases

1. **Volatility Clustering**: Model periods of high and low volatility
2. **Smile/Skew**: Capture volatility surface shape
3. **Exotic Options**: Path-dependent pricing with stochastic volatility
4. **Risk Management**: Scenario analysis with correlated dynamics
5. **Calibration**: Learn full market dynamics from option prices

## Advanced Topics

### Volatility State Interpretation

The state variable $Y_t$ is a **variance-like quantity**:
- If $Y_t = 0.04$, instantaneous volatility ≈ 20% annualized
- Network learns optimal mean reversion level and speed
- Correlation $\rho$ determines leverage effect strength

### Gradient Flow Through Coupled Dynamics

Backpropagation path:
1. Loss computed from Monte Carlo payoffs
2. Gradients flow back through averaging
3. Gradients flow through S paths
4. Gradients flow through coupled Y paths
5. Gradients couple at each time step (complex!)
6. Updates to f₁, f₂, f₃, f₄, ρ, Y₀

### Correlation Dynamics

The trained correlation $\rho$ captures:
- **Positive ρ**: Volatility and price move together
- **Negative ρ**: Volatility spikes when price drops (leverage effect)
- **|ρ|**: Strength of the relationship
- **Magnitude**: ~0.3-0.7 typical for equities

## Limitations and Considerations

1. **Computational Cost**: Slowest model due to 2D simulation
2. **Training Stability**: Complex coupled dynamics harder to train
3. **Overfitting Risk**: Many parameters (network + ρ + Y₀)
4. **Interpretability**: Less transparent than analytical models
5. **Hyperparameter Tuning**: Requires careful tuning of learning rate, dropout
6. **Memory Usage**: Stores 2D paths for all samples

## References

1. **Heston Model**: Classical stochastic volatility framework
2. **Neural SDEs**: Using neural networks for SDE coefficients
3. **Coupled Systems**: Joint optimization of interdependent dynamics
4. **Leverage Effect**: Empirical negative correlation in equity markets

## Code Example

```python
from semai.builtin_models import get_model_registry

# Get registry
registry = get_model_registry()

# Load 2D-NN model
model_2d = registry.get_model("2d_nn_standard")

# Train (most expensive model)
print("Training 2D-NN (this will take some time)...")
model_2d.train(X_train, y_train, X_val, y_val, verbose=True)

# Check learned parameters
print(f"Trained correlation ρ: {model_2d.training_history['final_rho']:.4f}")
print(f"Trained initial volatility Y₀: {model_2d.training_history['final_y0']:.6f}")

# Make predictions
prices = model_2d.predict(X_test)

# Evaluate
metrics = model_2d.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.6f}")
print(f"Test R²: {metrics['r2']:.6f}")

# Compare with other stochastic models
model_ids = ["sdenn_standard", "nnlv_standard", "2d_nn_standard"]
trained = [registry.get_trained_model(mid) for mid in model_ids]
for mid, model in zip(model_ids, trained):
    metrics = model.evaluate(X_test, y_test)
    print(f"{mid} RMSE: {metrics['rmse']:.6f}")
```

## Model Selection Guide

| Requirement | Model | Reason |
|-------------|-------|--------|
| Speed | Black-Scholes | Analytical |
| Balance | Deep Learning Net | Fast, reasonable accuracy |
| Vol clustering | 2D-NN | Explicitly models volatility state |
| Simplicity | NNLV | Easy to train |
| Accuracy | 2D-NN or SDENN | Most flexible |
| Fast training | NNLV | Fewer paths needed |
| Production | 2D-NN or SDENN | Best accuracy |
