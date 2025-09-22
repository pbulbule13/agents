"""
Euri AI Python SDK

A comprehensive Python SDK for the Euri AI API with integrations for popular frameworks.
"""

__version__ = "1.0.32"

# Core imports that should always work
try:
    from .client import EuriaiClient
except ImportError as e:
    print(f"Warning: Could not import EuriaiClient: {e}")
    EuriaiClient = None

try:
    from .embedding import EuriaiEmbeddingClient
    # Backward compatibility alias
    EuriaiEmbedding = EuriaiEmbeddingClient
except ImportError as e:
    print(f"Warning: Could not import EuriaiEmbeddingClient: {e}")
    EuriaiEmbeddingClient = None
    EuriaiEmbedding = None

# Main exports (only include what was successfully imported)
__all__ = []
if EuriaiClient is not None:
    __all__.append("EuriaiClient")
if EuriaiEmbeddingClient is not None:
    __all__.extend(["EuriaiEmbeddingClient", "EuriaiEmbedding"])


# Helper functions for optional dependencies
def check_optional_dependency(package_name: str, integration_name: str, install_extra: str = None) -> bool:
    """
    Check if an optional dependency is installed and provide helpful installation instructions.
    
    Args:
        package_name: The actual package name to import
        integration_name: The friendly name for the integration
        install_extra: The extras_require key for pip install euriai[extra]
    
    Returns:
        bool: True if package is available, False otherwise
        
    Raises:
        ImportError: With helpful installation instructions
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        extra_option = f"euriai[{install_extra}]" if install_extra else f"euriai[{integration_name.lower()}]"
        
        error_msg = (
            f"{integration_name} is not installed. Please install it using one of these methods:\n\n"
            f"Option 1 (Recommended): Install with euriai extras:\n"
            f"  pip install {extra_option}\n\n"
            f"Option 2: Install {integration_name} directly:\n"
            f"  pip install {package_name}\n\n"
            f"Option 3: Install all euriai integrations:\n"
            f"  pip install euriai[all]\n"
        )
        
        raise ImportError(error_msg)


def install_optional_dependency(package_name: str, integration_name: str, install_extra: str = None) -> bool:
    """
    Attempt to automatically install an optional dependency (USE WITH CAUTION).
    
    This function is provided for convenience but automatic installation can be risky.
    It's generally better to install dependencies manually.
    
    Args:
        package_name: The actual package name to install
        integration_name: The friendly name for the integration  
        install_extra: The extras_require key for pip install euriai[extra]
    
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    import subprocess
    import sys
    
    try:
        # Try to import first
        __import__(package_name)
        print(f"✓ {integration_name} is already installed")
        return True
    except ImportError:
        pass
    
    # Ask user for confirmation
    extra_option = f"euriai[{install_extra}]" if install_extra else f"euriai[{integration_name.lower()}]"
    
    print(f"🔍 {integration_name} is not installed.")
    print(f"📦 Recommended installation: pip install {extra_option}")
    
    response = input(f"Would you like to automatically install {package_name}? (y/N): ").lower()
    
    if response in ['y', 'yes']:
        try:
            print(f"📥 Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {integration_name} installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            print(f"💡 Try manually: pip install {extra_option}")
            return False
    else:
        print(f"💡 To install manually run: pip install {extra_option}")
        return False


# Lazy loading functions for optional integrations
def _get_langchain():
    """Lazy import for LangChain integration."""
    try:
        from . import langchain
        return langchain
    except ImportError:
        check_optional_dependency("langchain-core", "LangChain", "langchain")

def _get_crewai():
    """Lazy import for CrewAI integration."""
    try:
        from . import crewai
        return crewai
    except ImportError:
        check_optional_dependency("crewai", "CrewAI", "crewai")

def _get_autogen():
    """Lazy import for AutoGen integration."""
    try:
        from . import autogen
        return autogen
    except ImportError:
        check_optional_dependency("pyautogen", "AutoGen", "autogen")

def _get_smolagents():
    """Lazy import for SmolAgents integration."""
    try:
        from . import smolagents
        return smolagents
    except ImportError:
        check_optional_dependency("smolagents", "SmolAgents", "smolagents")

def _get_langgraph():
    """Lazy import for LangGraph integration."""
    try:
        from . import langgraph
        return langgraph
    except ImportError:
        check_optional_dependency("langgraph", "LangGraph", "langgraph")

def _get_llamaindex():
    """Lazy import for LlamaIndex integration."""
    try:
        from . import llamaindex
        return llamaindex
    except ImportError:
        check_optional_dependency("llama-index", "LlamaIndex", "llama-index")


# Create lazy loading properties
class _LazyLoader:
    """Lazy loader for optional integrations."""
    
    @property
    def langchain(self):
        return _get_langchain()
    
    @property
    def crewai(self):
        return _get_crewai()
    
    @property
    def autogen(self):
        return _get_autogen()
    
    @property
    def smolagents(self):
        return _get_smolagents()
    
    @property
    def langgraph(self):
        return _get_langgraph()
    
    @property
    def llamaindex(self):
        return _get_llamaindex()


# Create the lazy loader instance
_lazy = _LazyLoader()

# Make the integrations available as module-level attributes
def __getattr__(name: str):
    """Handle lazy loading of optional integrations."""
    if hasattr(_lazy, name):
        return getattr(_lazy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")