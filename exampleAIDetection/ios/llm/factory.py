# ios/llm/factory.py

from .model_config import PRIMARY_LLM, FAST_LLM

def get_default_llms():
    """
    Returns the primary and fast LLM models used in the search agent.
    """
    return PRIMARY_LLM, FAST_LLM