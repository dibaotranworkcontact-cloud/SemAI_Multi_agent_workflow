"""
Softmax Metrics Module
Implements softmax-based temperature scaling and token probability metrics
for balanced output generation and sampling variability control.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class SoftmaxMetrics:
    """
    Manages softmax metrics with temperature scaling for LLM output generation.
    Implements token probability distribution with configurable temperature.
    """
    
    def __init__(self, temperature: float = 0.2, top_k: int = 50, top_p: float = 0.9):
        """
        Initialize softmax metrics with temperature control.
        
        Args:
            temperature: Temperature for softmax scaling (0.2 = cold/deterministic, high = random)
            top_k: Keep only top K tokens for sampling
            top_p: Cumulative probability threshold for nucleus sampling
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.metrics_history = []
    
    def apply_temperature_scaling(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        Lower temperature makes distribution more peaked (prioritizes high-scoring tokens).
        Higher temperature makes distribution more uniform (random).
        
        Args:
            logits: Raw logits from model (typically shape [vocab_size])
        
        Returns:
            Temperature-scaled logits
        """
        scaled_logits = logits / self.temperature
        return scaled_logits
    
    def compute_softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities with numerical stability.
        
        Args:
            logits: Input logits (can be raw or temperature-scaled)
        
        Returns:
            Probability distribution (sums to 1)
        """
        # Subtract max for numerical stability
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits)
        return probabilities
    
    def get_high_confidence_tokens(self, logits: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k high-confidence tokens with their probabilities.
        
        Args:
            logits: Input logits
            k: Number of top tokens (uses self.top_k if None)
        
        Returns:
            Tuple of (token_indices, probabilities) sorted by probability descending
        """
        if k is None:
            k = self.top_k
        
        # Apply temperature scaling
        scaled_logits = self.apply_temperature_scaling(logits)
        
        # Compute softmax probabilities
        probs = self.compute_softmax(scaled_logits)
        
        # Get top-k indices
        top_k_indices = np.argsort(-probs)[:k]
        top_k_probs = probs[top_k_indices]
        
        return top_k_indices, top_k_probs
    
    def nucleus_sampling(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement nucleus (top-p) sampling for controlled diversity.
        Selects smallest set of tokens with cumulative probability >= top_p.
        
        Args:
            logits: Input logits
        
        Returns:
            Tuple of (token_indices, normalized_probabilities)
        """
        # Apply temperature scaling
        scaled_logits = self.apply_temperature_scaling(logits)
        
        # Compute softmax probabilities
        probs = self.compute_softmax(scaled_logits)
        
        # Sort by probability descending
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        
        # Compute cumulative probabilities
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Find tokens within top_p threshold
        nucleus_mask = cumsum_probs <= self.top_p
        # Always include at least the top token
        nucleus_mask[0] = True
        
        nucleus_indices = sorted_indices[nucleus_mask]
        nucleus_probs = sorted_probs[nucleus_mask]
        
        # Renormalize to sum to 1
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
        
        return nucleus_indices, nucleus_probs
    
    def compute_entropy(self, logits: np.ndarray) -> float:
        """
        Compute Shannon entropy of probability distribution.
        Indicates output variability (higher = more random, lower = more deterministic).
        
        Args:
            logits: Input logits
        
        Returns:
            Entropy value (0 = deterministic, increases with randomness)
        """
        probs = self.compute_softmax(self.apply_temperature_scaling(logits))
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def compute_confidence(self, logits: np.ndarray) -> float:
        """
        Compute model confidence (max probability in distribution).
        Higher values indicate stronger preference for top token.
        
        Args:
            logits: Input logits
        
        Returns:
            Confidence score (0-1, higher = more confident)
        """
        probs = self.compute_softmax(self.apply_temperature_scaling(logits))
        confidence = float(np.max(probs))
        return confidence
    
    def compute_gini_coefficient(self, logits: np.ndarray) -> float:
        """
        Compute Gini coefficient (measure of probability distribution inequality).
        Values closer to 0 = uniform distribution
        Values closer to 1 = concentrated distribution (one token dominates)
        
        Args:
            logits: Input logits
        
        Returns:
            Gini coefficient (0-1)
        """
        probs = self.compute_softmax(self.apply_temperature_scaling(logits))
        
        # Sort probabilities
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        
        # Gini formula: 1 - (2 * sum of weighted ranks) / (n * sum of values)
        cumsum = np.cumsum(sorted_probs)
        gini = 1 - (2 * np.sum(cumsum) / (n * np.sum(probs)))
        
        return float(gini)
    
    def adjust_temperature_for_variability(self, current_entropy: float, target_entropy: float) -> float:
        """
        Dynamically adjust temperature to achieve target entropy.
        
        Args:
            current_entropy: Current entropy of distribution
            target_entropy: Desired entropy level
        
        Returns:
            Adjusted temperature value
        """
        if current_entropy < target_entropy:
            # Increase temperature for more randomness
            self.temperature = min(2.0, self.temperature * 1.1)
        else:
            # Decrease temperature for more determinism
            self.temperature = max(0.1, self.temperature * 0.9)
        
        return self.temperature
    
    def get_sampling_distribution(self, logits: np.ndarray, method: str = "nucleus") -> np.ndarray:
        """
        Get final sampling distribution based on selected method.
        
        Args:
            logits: Input logits
            method: "top_k", "nucleus", or "softmax"
        
        Returns:
            Probability distribution for sampling
        """
        if method == "top_k":
            _, probs = self.get_high_confidence_tokens(logits)
            # Create full distribution with zeros for non-top-k tokens
            full_dist = np.zeros(len(logits))
            top_k_indices, top_k_probs = self.get_high_confidence_tokens(logits)
            full_dist[top_k_indices] = top_k_probs
            return full_dist
        
        elif method == "nucleus":
            nucleus_indices, nucleus_probs = self.nucleus_sampling(logits)
            full_dist = np.zeros(len(logits))
            full_dist[nucleus_indices] = nucleus_probs
            return full_dist
        
        elif method == "softmax":
            scaled_logits = self.apply_temperature_scaling(logits)
            return self.compute_softmax(scaled_logits)
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def record_metrics(self, logits: np.ndarray, token_id: Optional[int] = None, 
                      metadata: Optional[Dict] = None) -> Dict:
        """
        Record comprehensive metrics for a token prediction.
        
        Args:
            logits: Input logits
            token_id: Selected token ID (optional)
            metadata: Additional metadata (optional)
        
        Returns:
            Dictionary of recorded metrics
        """
        metrics = {
            "temperature": self.temperature,
            "entropy": self.compute_entropy(logits),
            "confidence": self.compute_confidence(logits),
            "gini": self.compute_gini_coefficient(logits),
            "token_id": token_id,
            "metadata": metadata or {}
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_metrics_summary(self) -> Dict:
        """
        Get summary statistics from metrics history.
        
        Returns:
            Dictionary with average and aggregate metrics
        """
        if not self.metrics_history:
            return {}
        
        entropies = [m["entropy"] for m in self.metrics_history]
        confidences = [m["confidence"] for m in self.metrics_history]
        ginis = [m["gini"] for m in self.metrics_history]
        
        return {
            "avg_entropy": float(np.mean(entropies)),
            "avg_confidence": float(np.mean(confidences)),
            "avg_gini": float(np.mean(ginis)),
            "max_entropy": float(np.max(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_confidence": float(np.max(confidences)),
            "min_confidence": float(np.min(confidences)),
            "total_predictions": len(self.metrics_history),
            "current_temperature": self.temperature
        }
    
    def reset_history(self):
        """Clear metrics history."""
        self.metrics_history = []


class TokenSamplingController:
    """
    Controls token sampling variability with softmax metrics.
    Provides deterministic high-scoring token priority with configurable randomness.
    """
    
    def __init__(self, temperature: float = 0.2, seed: Optional[int] = None):
        """
        Initialize token sampling controller.
        
        Args:
            temperature: Base temperature for softmax (0.2 = balanced, prioritizes high scores)
            seed: Random seed for reproducibility
        """
        self.softmax_metrics = SoftmaxMetrics(temperature=temperature)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def sample_token(self, logits: np.ndarray, method: str = "nucleus") -> int:
        """
        Sample a token from logits using specified method.
        
        Args:
            logits: Model logits
            method: "top_k", "nucleus", or "softmax"
        
        Returns:
            Sampled token ID
        """
        distribution = self.softmax_metrics.get_sampling_distribution(logits, method=method)
        token_id = np.random.choice(len(logits), p=distribution)
        
        # Record metrics
        self.softmax_metrics.record_metrics(logits, token_id=token_id)
        
        return int(token_id)
    
    def get_top_tokens(self, logits: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k tokens with their probabilities (for display/analysis).
        
        Args:
            logits: Model logits
            k: Number of top tokens to return
        
        Returns:
            List of (token_id, probability) tuples
        """
        indices, probs = self.softmax_metrics.get_high_confidence_tokens(logits, k=k)
        return list(zip(indices, probs))
    
    def get_current_config(self) -> Dict:
        """Get current sampling configuration."""
        return {
            "temperature": self.softmax_metrics.temperature,
            "top_k": self.softmax_metrics.top_k,
            "top_p": self.softmax_metrics.top_p,
            "seed": self.seed
        }


# Global softmax metrics instance for system-wide use
_global_softmax_metrics = None

def get_global_softmax_metrics(temperature: float = 0.2) -> SoftmaxMetrics:
    """
    Get or create global softmax metrics instance.
    
    Args:
        temperature: Temperature for softmax (default 0.2 for balanced output)
    
    Returns:
        Global SoftmaxMetrics instance
    """
    global _global_softmax_metrics
    if _global_softmax_metrics is None:
        _global_softmax_metrics = SoftmaxMetrics(temperature=temperature)
    return _global_softmax_metrics

def reset_global_softmax_metrics():
    """Reset global softmax metrics instance."""
    global _global_softmax_metrics
    _global_softmax_metrics = None
