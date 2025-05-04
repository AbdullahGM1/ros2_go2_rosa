#!/usr/bin/env python3
"""
LLM configuration for the Unitree Go2 ROSA Agent.
This module handles all Large Language Model related configurations and initialization.
"""

from langchain_ollama import ChatOllama


def initialize_llm(model_name: str) -> ChatOllama:
    """
    Initialize and configure the local LLM for the Go2 agent.
    
    Args:
        model_name (str): Name of the LLM model to use (e.g., 'qwen3:8b')
        
    Returns:
        ChatOllama: Configured LLM instance
    """
    return ChatOllama(
        model=model_name,
        temperature=0.0,
        max_retries=2,
        num_ctx=8192,
    )