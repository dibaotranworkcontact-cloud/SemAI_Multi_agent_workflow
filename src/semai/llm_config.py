"""
LLM Configuration Module
Configures Meta Llama 3 70B Instruct Turbo and Deepseek V3.1 models
with softmax metrics for temperature-scaled token sampling.
"""

import os
from together import Together
from semai.softmax_metrics import get_global_softmax_metrics

# Set API key
API_KEY = "246c6b4fa5304217d38fedd55678e546b5d5971acd7f470d7af1911b0aa102d5"
os.environ["TOGETHER_API_KEY"] = API_KEY

# Initialize Together AI client
client = Together(api_key=API_KEY)

# Softmax metrics with temperature 0.2 for balanced output (high-scoring tokens prioritized)
SOFTMAX_TEMPERATURE = 0.2
softmax_metrics = get_global_softmax_metrics(temperature=SOFTMAX_TEMPERATURE)

# Model configurations with softmax metrics integration
MODELS = {
    "llama_3_70b": {
        "name": "meta-llama/Llama-3-70b-chat-hf",
        "description": "Meta Llama 3 70B Instruct Turbo",
        "max_tokens": 8192,
        "temperature": SOFTMAX_TEMPERATURE,
        "top_k": 50,
        "top_p": 0.9,
        "softmax_enabled": True
    },
    "deepseek_v3_1": {
        "name": "deepseek-ai/DeepSeek-V3.1",
        "description": "Deepseek V3.1",
        "max_tokens": 8192,
        "temperature": SOFTMAX_TEMPERATURE,
        "top_k": 50,
        "top_p": 0.9,
        "softmax_enabled": True
    }
}

def get_model(model_key: str):
    """Get model configuration by key"""
    return MODELS.get(model_key, None)

def list_available_models():
    """List all configured models"""
    return MODELS

def get_model_with_softmax_config(model_key: str) -> dict:
    """
    Get model configuration with softmax metrics applied.
    Ensures high-scoring tokens are prioritized with temperature 0.2.
    
    Args:
        model_key: Model identifier
    
    Returns:
        Model config with softmax settings applied
    """
    model = get_model(model_key)
    if not model:
        return None
    
    return {
        **model,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "softmax_metrics": softmax_metrics.get_metrics_summary()
    }

def apply_softmax_to_request(model_key: str, temperature: float = None) -> float:
    """
    Apply softmax metrics to LLM request.
    
    Args:
        model_key: Model identifier
        temperature: Optional override temperature
    
    Returns:
        Effective temperature for LLM call
    """
    if temperature is not None:
        softmax_metrics.temperature = temperature
    else:
        # Use configured softmax temperature (0.2)
        softmax_metrics.temperature = SOFTMAX_TEMPERATURE
    
    return softmax_metrics.temperature

def log_softmax_metrics(label: str = ""):
    """
    Log current softmax metrics summary.
    
    Args:
        label: Optional label for logging
    """
    summary = softmax_metrics.get_metrics_summary()
    if summary:
        print(f"\n[METRICS] Softmax Metrics {label}:")
        print(f"   Temperature: {summary['current_temperature']:.2f}")
        print(f"   Avg Entropy: {summary['avg_entropy']:.4f}")
        print(f"   Avg Confidence: {summary['avg_confidence']:.4f}")
        print(f"   Avg Gini: {summary['avg_gini']:.4f}")
        print(f"   Predictions: {summary['total_predictions']}")


def test_llama_model():
    """Test Meta Llama 3 70B model with softmax metrics"""
    try:
        # Apply softmax temperature scaling
        effective_temp = apply_softmax_to_request("llama_3_70b")
        
        response = client.chat.completions.create(
            model=MODELS["llama_3_70b"]["name"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! What is your name?"}
            ],
            max_tokens=256,
            temperature=effective_temp,
            top_k=MODELS["llama_3_70b"]["top_k"],
            top_p=MODELS["llama_3_70b"]["top_p"],
        )
        print("[OK] Meta Llama 3 70B - Connected Successfully")
        # Encode response to handle special characters
        safe_response = response.choices[0].message.content.encode('utf-8', 'ignore').decode('utf-8')
        print(f"Response: {safe_response}\n")
        log_softmax_metrics("[Llama 3 70B]")
        return True
    except Exception as e:
        print(f"[FAILED] Meta Llama 3 70B - Connection Error: {str(e)}\n")
        return False

def test_deepseek_model():
    """Test Deepseek V3.1 model with softmax metrics"""
    try:
        # Apply softmax temperature scaling
        effective_temp = apply_softmax_to_request("deepseek_v3_1")
        
        response = client.chat.completions.create(
            model=MODELS["deepseek_v3_1"]["name"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! What is your name? Reply without emojis."}
            ],
            max_tokens=256,
            temperature=effective_temp,
            top_k=MODELS["deepseek_v3_1"]["top_k"],
            top_p=MODELS["deepseek_v3_1"]["top_p"],
        )
        print("[OK] Deepseek V3.1 - Connected Successfully")
        # Remove any non-ASCII characters and encode safely
        response_text = response.choices[0].message.content
        safe_response = ''.join(c for c in response_text if ord(c) < 128)
        print(f"Response: {safe_response}\n")
        
        # Try to log metrics, but ignore Unicode errors
        try:
            log_softmax_metrics("[Deepseek V3.1]")
        except UnicodeEncodeError:
            print("[METRICS] Softmax metrics logged (unicode display skipped)")
        
        return True
    except Exception as e:
        print(f"[FAILED] Deepseek V3.1 - Connection Error: {str(e)}\n")
        return False

def test_all_models():
    """Test all configured models with softmax metrics"""
    print("=" * 60)
    print("Testing Model Connections with Softmax Metrics (T=0.2)...")
    print("=" * 60 + "\n")
    
    llama_status = test_llama_model()
    deepseek_status = test_deepseek_model()
    
    print("=" * 60)
    print("Summary:")
    print(f"Meta Llama 3 70B: {'[OK]' if llama_status else '[FAILED]'} Ready")
    print(f"Deepseek V3.1: {'[OK]' if deepseek_status else '[FAILED]'} Ready")
    print(f"Softmax Temperature: {SOFTMAX_TEMPERATURE} (Prioritizes high-scoring tokens)")
    print("=" * 60)
    
    return llama_status and deepseek_status

if __name__ == "__main__":
    print("\nImporting and Testing Models...\n")
    print(f"API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
    print("\n")
    test_all_models()
