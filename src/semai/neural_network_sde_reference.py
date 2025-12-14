"""
Neural Network SDE - Reference Information
============================================

This file contains theoretical and implementation reference for the 
Neural Network SDE (Stochastic Differential Equation) approach for 
European option pricing.

SECTION 2.1: OPTIMIZATION FOR EUROPEAN OPTIONS
===============================================

Monte Carlo Approximation:
The price P_i(θ) of the i-th financial derivative can be approximated via 
Monte Carlo simulation of the SDEs (S_t, Y_t):

    P_i(θ) ≈ P_i^L(θ) = e^(-rT) * (1/L) * Σ(ℓ=1 to L) g_i(S_T^ℓ)     [Eq. 5]

where:
    - (S_t^ℓ, Y_t^ℓ) are i.i.d. Monte Carlo paths of the SDE
    - T_i = T (notational convenience)
    - g_i is the payoff function
    - r is the risk-free rate
    - L is the number of Monte Carlo samples

Objective Function:
The objective function J(θ) is approximated as:

    J_L(θ) = (1/N) * Σ(i=1 to N) (P_i^market - P_i^L(θ))²     [Eq. 6]

where:
    - N is the number of contracts
    - P_i^market is the market price
    - P_i^L(θ) is the model price approximation

Gradient Estimation (Naive Approach - BIASED):
The naive gradient estimate is:

    ∇_θ J_L(θ) = -2/N * Σ(i=1 to N) (P_i^market - P_i^L(θ)) * ∇_θ P_i^L(θ)     [Eq. 7]

Issue: This is a BIASED estimate because E[∇_θ J_L(θ)] ≠ ∇_θ J(θ)

True Gradient (Unbiased):
The true direction of steepest descent is:

    ∇_θ J(θ) = -1/N * Σ(i=1 to N) (P_i^market - E[e^(-rT) * g_i(S_T)])
                × ∇_θ [e^(-rT) * E[g_i(S_T)]]     [Eq. 8]

Unbiased Monte Carlo Gradient (CORRECTED):
To obtain an unbiased estimate, use independent Monte Carlo samples:

    G_L(θ) = -2/N * Σ(i=1 to N) (P_i^market - e^(-rT) * (1/L) * Σ(ℓ=1 to L) g_i(S_T^ℓ))
             × ∇_θ [e^(-rT) * (1/L) * Σ(ℓ=L+1 to 2L) g_i(S_T^ℓ)]     [Eq. 9]

Key Innovation:
The Monte Carlo samples for the gradient computation (S_t^(L+1), ..., S_t^(2L)) 
are INDEPENDENT of the samples for the price approximation (S_t^1, ..., S_t^L).

This ensures: E[G_L(θ)] = ∇_θ J(θ)     [Eq. 10]

Stochastic Gradient Descent Algorithm:
1. Generate 2L Monte Carlo samples of SDE paths (S_t, Y_t) for parameters θ^(k)
2. Calculate G_L(θ^(k)) using the Monte Carlo samples
3. Update parameters:
   θ^(k+1) = θ^(k) - α^(k) * G_L(θ^(k))     [Eq. 11]
   where α^(k) is the learning rate

Computational Efficiency:
All contracts (i=1,2,...,N) share the SAME set of Monte Carlo paths.
This saves computational cost compared to separately simulating 2L paths 
for each contract (which would require N×2L total paths per iteration).


SECTION 2.2: PDE APPROACH TO OPTIMIZATION
==========================================

General Objective Function:
    J(θ) = 1/N * Σ(i=1 to N) ℓ(P_i^market, P_i(θ))     [Eq. 12]

where ℓ(z,v) is a general loss function.

Kolmogorov PDE:
The model-generated price P_i(θ) can be evaluated using:

    -∂v_i/∂t = μ(x,y;θ) * ∂v_i/∂x + μ_Y(x,y;θ) * ∂v_i/∂y
                + 1/2 * σ(x,y;θ)² * ∂²v_i/∂x² 
                + ρ * σ(x,y;θ) * σ_Y(x,y;θ) * ∂²v_i/(∂x∂y)
                + 1/2 * σ_Y(x,y;θ)² * ∂²v_i/∂y² - r*v     [Eq. 13]

where:
    - μ(x,y;θ) is the drift of the stock price
    - μ_Y(x,y;θ) is the drift of the stochastic volatility
    - σ(x,y;θ) is the diffusion of the stock price
    - σ_Y(x,y;θ) is the diffusion of the stochastic volatility
    - ρ is the correlation between S and Y
    - r is the risk-free rate
    - θ are the model parameters

Boundary Condition:
    v_i(T, s, y) = g_i(s)

Solution:
    P_i(θ) = v_i(t=0, s_0, y_0)

Implementation:
1. Numerically solve PDE (13) using finite-difference methods
2. Optimize over parameters using automatic differentiation
3. Minimize objective function (12)

Advantages of PDE Approach:
- Can optimize over GENERAL loss functions
- Necessary for American and Bermudan options (allow early exercise)
- More accurate for path-dependent options
- Handles nonlinear option pricing

Limitations:
- Computationally more expensive
- Requires sophisticated numerical PDE solvers
- More complex implementation


KEY PARAMETERS FOR NEURAL NETWORK SDE
======================================

Monte Carlo Configuration:
- L: Number of Monte Carlo samples per path
- 2L: Total samples needed (L for price, L for gradient)
- N: Number of contracts

SDE Parameters:
- μ(·): Stock price drift function
- σ(·): Stock price volatility function
- μ_Y(·): Stochastic volatility drift
- σ_Y(·): Stochastic volatility diffusion
- ρ: Correlation between S and Y
- r: Risk-free rate

Training Parameters:
- α^(k): Learning rate (adaptive)
- θ: Neural network parameters
- Loss function: ℓ(·,·) (e.g., MSE for European options)

Numerical Parameters:
- Time steps: Discretization steps for SDE simulation
- dt: Time increment for Euler scheme
- Grid resolution: For PDE solving (if using PDE approach)


REFERENCES
==========

This implementation is based on:
- Neural Networks for Option Pricing and Hedging
- Using SDEs for calibration of financial models
- Monte Carlo methods for derivative pricing
- PDE methods for option valuation under stochastic volatility

The key insight is using independent Monte Carlo samples to ensure 
unbiased gradient estimation, enabling efficient parameter optimization.
"""
