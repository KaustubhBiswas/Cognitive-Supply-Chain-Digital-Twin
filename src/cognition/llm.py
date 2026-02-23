"""
LLM Configuration and Initialization

Provides flexible LLM initialization for the cognition module.
Supports Groq (cloud) and Ollama (local) backends.
"""

import logging
import os
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_PROVIDER = "groq"  # "groq" or "ollama"
DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"
DEFAULT_OLLAMA_MODEL = "llama3:8b"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 512


class LLMConfig:
    """Configuration for LLM initialization."""
    
    def __init__(
        self,
        provider: Literal["groq", "ollama"] = DEFAULT_PROVIDER,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM configuration.
        
        Args:
            provider: "groq" for cloud API or "ollama" for local
            model: Model name (defaults based on provider)
            api_key: API key for Groq (reads GROQ_API_KEY env var if not provided)
            temperature: Response randomness (0 = deterministic)
            max_tokens: Maximum tokens in response
            base_url: Custom base URL (for Ollama: default http://localhost:11434)
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # Set model based on provider if not specified
        if model is None:
            self.model = DEFAULT_GROQ_MODEL if provider == "groq" else DEFAULT_OLLAMA_MODEL
        else:
            self.model = model
        
        # Get API key from env if not provided
        if api_key is None and provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY", "")
        else:
            self.api_key = api_key or ""


def create_llm(
    config: Optional[LLMConfig] = None,
    provider: Literal["groq", "ollama"] = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    test_connection: bool = True,
) -> Optional[Any]:
    """
    Create and initialize an LLM instance.
    
    Args:
        config: LLMConfig instance (if provided, other args are ignored)
        provider: "groq" for cloud API or "ollama" for local
        model: Model name
        api_key: API key for Groq
        temperature: Response randomness
        max_tokens: Maximum tokens in response
        test_connection: Whether to test the connection before returning
        
    Returns:
        LangChain LLM instance or None if initialization fails
        
    Usage:
        # Using Groq (recommended)
        llm = create_llm(provider="groq", api_key="gsk_...")
        
        # Using Ollama (local)
        llm = create_llm(provider="ollama", model="llama3:8b")
        
        # Using config object
        config = LLMConfig(provider="groq", model="llama-3.1-70b-versatile")
        llm = create_llm(config=config)
    """
    if config is None:
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    try:
        if config.provider == "groq":
            return _create_groq_llm(config, test_connection)
        elif config.provider == "ollama":
            return _create_ollama_llm(config, test_connection)
        else:
            logger.error(f"Unknown provider: {config.provider}")
            return None
    except Exception as e:
        logger.warning(f"Failed to create LLM: {e}")
        return None


def _create_groq_llm(config: LLMConfig, test_connection: bool) -> Optional[Any]:
    """Create Groq LLM instance."""
    if not config.api_key:
        logger.warning("GROQ_API_KEY not set. Get your free key at: https://console.groq.com")
        return None
    
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        logger.error("langchain-groq not installed. Run: pip install langchain-groq")
        return None
    
    llm = ChatGroq(
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    
    if test_connection:
        try:
            llm.invoke("test")
            logger.info(f"Connected to Groq: {config.model}")
        except Exception as e:
            logger.error(f"Groq connection failed: {e}")
            return None
    
    return llm


def _create_ollama_llm(config: LLMConfig, test_connection: bool) -> Optional[Any]:
    """Create Ollama LLM instance."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.error("langchain-ollama not installed. Run: pip install langchain-ollama")
        return None
    
    kwargs = {
        "model": config.model,
        "temperature": config.temperature,
        "num_predict": config.max_tokens,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    
    llm = ChatOllama(**kwargs)
    
    if test_connection:
        try:
            llm.invoke("test")
            logger.info(f"Connected to Ollama: {config.model}")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return None
    
    return llm


# Convenience functions
def create_groq_llm(
    api_key: Optional[str] = None,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Optional[Any]:
    """
    Create a Groq LLM instance (convenience function).
    
    Args:
        api_key: Groq API key (or set GROQ_API_KEY env var)
        model: Model name (default: llama-3.1-70b-versatile)
        temperature: Response randomness (default: 0)
        
    Returns:
        ChatGroq instance or None
    """
    return create_llm(
        provider="groq",
        api_key=api_key,
        model=model,
        temperature=temperature,
    )


def create_ollama_llm(
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Optional[Any]:
    """
    Create an Ollama LLM instance (convenience function).
    
    Args:
        model: Model name (default: llama3:8b)
        temperature: Response randomness (default: 0)
        
    Returns:
        ChatOllama instance or None
    """
    return create_llm(
        provider="ollama",
        model=model,
        temperature=temperature,
    )


# Available models reference
GROQ_MODELS = {
    "llama-3.1-70b-versatile": "Best accuracy, 70B parameters",
    "llama-3.1-8b-instant": "Fast, 8B parameters",
    "llama-3.2-90b-vision-preview": "Vision capable, 90B parameters",
    "mixtral-8x7b-32768": "Good balance, 32K context",
    "gemma2-9b-it": "Google's Gemma 2, 9B parameters",
}

OLLAMA_MODELS = {
    "llama3.1:70b": "Best accuracy, requires ~40GB RAM",
    "llama3.1:8b": "Good balance, requires ~8GB RAM",
    "llama3:8b": "Fast and capable, requires ~8GB RAM",
    "llama3.2:3b": "Very fast, requires ~4GB RAM",
    "mistral:7b": "Efficient, requires ~8GB RAM",
}
