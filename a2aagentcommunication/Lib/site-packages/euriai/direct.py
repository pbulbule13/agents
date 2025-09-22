"""
Direct Import Module for Euri AI SDK

This module provides direct access to integrations without going through the main __init__.py,
which can be useful if there are import issues with the main package.

Usage:
    # Instead of: from euriai.langchain import EuriaiChatModel
    # Use: from euriai.direct import langchain_chat_model
    
    from euriai.direct import langchain_chat_model, langchain_embeddings
    
    chat_model = langchain_chat_model(
        api_key="your-key",
        model="gpt-4.1-nano"
    )
"""

# Direct imports that bypass __init__.py
def langchain_chat_model(api_key: str, model: str = "gpt-4.1-nano", **kwargs):
    """Create a LangChain chat model directly."""
    from .langchain import EuriaiChatModel
    return EuriaiChatModel(api_key=api_key, model=model, **kwargs)

def langchain_llm(api_key: str, model: str = "gpt-4.1-nano", **kwargs):
    """Create a LangChain LLM directly."""
    from .langchain import EuriaiLLM
    return EuriaiLLM(api_key=api_key, model=model, **kwargs)

def langchain_embeddings(api_key: str, model: str = "text-embedding-3-small", **kwargs):
    """Create LangChain embeddings directly."""
    from .langchain import EuriaiEmbeddings
    return EuriaiEmbeddings(api_key=api_key, model=model, **kwargs)

def autogen_instance(api_key: str, default_model: str = "gpt-4.1-nano"):
    """Create an AutoGen instance directly."""
    from .autogen import EuriaiAutoGen
    return EuriaiAutoGen(api_key=api_key, default_model=default_model)

def crewai_instance(api_key: str, default_model: str = "gpt-4.1-nano", **kwargs):
    """Create a CrewAI instance directly."""
    from .crewai import EuriaiCrewAI
    return EuriaiCrewAI(api_key=api_key, default_model=default_model, **kwargs)

def client(api_key: str, model: str = "gpt-4.1-nano", **kwargs):
    """Create a basic Euri client directly."""
    from .client import EuriaiClient
    return EuriaiClient(api_key=api_key, model=model, **kwargs)

def embedding_client(api_key: str, model: str = "text-embedding-3-small"):
    """Create an embedding client directly."""
    from .embedding import EuriaiEmbeddingClient
    return EuriaiEmbeddingClient(api_key=api_key, model=model)

# Class exports for direct import
def get_langchain_classes():
    """Get LangChain classes for direct import."""
    from .langchain import EuriaiChatModel, EuriaiLLM, EuriaiEmbeddings
    return {
        'EuriaiChatModel': EuriaiChatModel,
        'EuriaiLLM': EuriaiLLM,
        'EuriaiEmbeddings': EuriaiEmbeddings
    }

def get_autogen_classes():
    """Get AutoGen classes for direct import."""
    from .autogen import EuriaiAutoGen
    return {'EuriaiAutoGen': EuriaiAutoGen}

def get_crewai_classes():
    """Get CrewAI classes for direct import."""
    from .crewai import EuriaiCrewAI
    return {'EuriaiCrewAI': EuriaiCrewAI}

def get_core_classes():
    """Get core classes for direct import."""
    from .client import EuriaiClient
    from .embedding import EuriaiEmbeddingClient
    return {
        'EuriaiClient': EuriaiClient,
        'EuriaiEmbeddingClient': EuriaiEmbeddingClient
    } 