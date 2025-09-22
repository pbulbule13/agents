from typing import Optional, Dict, Any, List, Union, Callable
from types import SimpleNamespace
from euriai.client import EuriaiClient
import asyncio

# AutoGen Import Strategy: Try modern first, then legacy
AUTOGEN_VERSION = "unknown"
AUTOGEN_AGENTCHAT_AVAILABLE = False
AUTOGEN_CORE_AVAILABLE = False

# Initialize all variables to None first
autogen = agentchat = autogen_core = None
AssistantAgent = UserProxyAgent = RoundRobinGroupChat = SelectorGroupChat = None
GroupChat = GroupChatManager = None
TextMessage = MultiModalMessage = MaxMessageTermination = TextMentionTermination = None
ChatCompletionClient = MessageContext = None

try:
    # Try AutoGen v0.4+ AgentChat first
    try:
        from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
        from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
        from autogen_agentchat.messages import TextMessage, MultiModalMessage
        from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
        import autogen_agentchat as agentchat
        AUTOGEN_AGENTCHAT_AVAILABLE = True
        AUTOGEN_VERSION = "agentchat_v0.4+"
    except ImportError:
        pass
    
    # Try AutoGen Core for advanced patterns
    try:
        import autogen_core
        from autogen_core.components.models import ChatCompletionClient
        from autogen_core.base import MessageContext
        AUTOGEN_CORE_AVAILABLE = True
        if AUTOGEN_VERSION == "unknown":
            AUTOGEN_VERSION = "core_v0.4+"
    except ImportError:
        pass
    
    # Fallback to legacy AutoGen
    if not AUTOGEN_AGENTCHAT_AVAILABLE and not AUTOGEN_CORE_AVAILABLE:
        try:
            import autogen
            from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
            AUTOGEN_VERSION = "legacy_v0.2"
        except ImportError:
            pass

except ImportError:
    # All AutoGen imports failed - variables already set to None above
    pass

def _ensure_autogen_available():
    """Ensure at least one AutoGen version is available."""
    global autogen
    if not any([AUTOGEN_AGENTCHAT_AVAILABLE, AUTOGEN_CORE_AVAILABLE, autogen]):
        from . import check_optional_dependency
        check_optional_dependency("autogen-agentchat", "AutoGen AgentChat", "autogen")

class EuriaiModelClient:
    """
    Enhanced model client that uses Euri API for AutoGen integration.
    Compatible with both legacy and modern AutoGen versions.
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initialize the Euri model client.
        
        Args:
            config: Configuration dictionary containing:
                - model: Euri model name (e.g., 'gpt-4.1-nano', 'claude-3-5-sonnet')
                - api_key: Euri API key
                - temperature: Model temperature (optional)
                - max_tokens: Maximum tokens (optional)
        """
        self.config = config
        self.model = config["model"]
        self.api_key = config.get("api_key")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        
        if not self.api_key:
            raise ValueError("Euri API key is required in config")
        
        # Initialize Euri client
        self.client = EuriaiClient(
            api_key=self.api_key,
            model=self.model
        )
    
    def create(self, params: Dict[str, Any]) -> SimpleNamespace:
        """
        Create a response using the Euri API.
        Compatible with both legacy and modern AutoGen.
        
        Args:
            params: Parameters containing:
                - messages: List of message dictionaries or objects
                - n: Number of responses (default 1)
                - temperature: Temperature override
                - max_tokens: Max tokens override
                
        Returns:
            Response object following AutoGen's ModelClientResponseProtocol
        """
        # Extract parameters
        messages = params.get("messages", [])
        n = params.get("n", 1)
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_tokens)
        
        # Convert messages to prompt format (handle both old and new message formats)
        prompt = self._convert_messages_to_prompt(messages)
        
        # Create response object
        response = SimpleNamespace()
        response.choices = []
        response.model = self.model
        response.usage = SimpleNamespace()
        
        # Generate responses
        for _ in range(n):
            try:
                # Call Euri API
                euri_response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract content
                content = euri_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Create choice object
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = content
                choice.message.function_call = None
                choice.finish_reason = "stop"
                
                response.choices.append(choice)
                
                # Add usage info if available
                if "usage" in euri_response:
                    usage = euri_response["usage"]
                    response.usage.prompt_tokens = usage.get("prompt_tokens", 0)
                    response.usage.completion_tokens = usage.get("completion_tokens", 0)
                    response.usage.total_tokens = usage.get("total_tokens", 0)
                
            except Exception as e:
                # Create error response
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = f"I apologize, but I'm having trouble generating a response right now."
                choice.message.function_call = None
                choice.finish_reason = "error"
                response.choices.append(choice)
        
        return response
    
    # AsyncChatCompletionClient interface for AutoGen Core
    async def create_async(self, params: Dict[str, Any]) -> SimpleNamespace:
        """Async version for AutoGen Core compatibility."""
        return self.create(params)
    
    def message_retrieval(self, response: SimpleNamespace) -> List[str]:
        """Retrieve messages from the response."""
        return [choice.message.content for choice in response.choices]
    
    def cost(self, response: SimpleNamespace) -> float:
        """Calculate the cost of the response."""
        return 0.0
    
    @staticmethod
    def get_usage(response: SimpleNamespace) -> Dict[str, Any]:
        """Get usage statistics from the response."""
        usage = getattr(response, 'usage', SimpleNamespace())
        return {
            "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(usage, 'completion_tokens', 0),
            "total_tokens": getattr(usage, 'total_tokens', 0),
            "cost": 0.0,
            "model": response.model
        }
    
    def _convert_messages_to_prompt(self, messages: List[Any]) -> str:
        """
        Convert AutoGen messages to a prompt string.
        Handles both legacy dict format and modern message objects.
        
        Args:
            messages: List of message dictionaries or objects
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            # Handle modern AutoGen message objects
            if hasattr(message, 'content') and hasattr(message, 'source'):
                role = getattr(message.source, 'name', 'user') if hasattr(message.source, 'name') else 'user'
                content = message.content
            # Handle legacy dict format
            elif isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
            else:
                # Fallback for unknown format
                role = "user"
                content = str(message)
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")
        
        return "\n".join(prompt_parts)


class EuriaiAutoGen:
    """
    Enhanced AutoGen integration supporting both modern v0.4+ and legacy versions.
    Provides unified interface for AgentChat Teams, Core patterns, and legacy GroupChat.
    """
    
    def __init__(self, api_key: str, default_model: str = "gpt-4.1-nano", debug: bool = False):
        """
        Initialize the EuriaiAutoGen wrapper.
        
        Args:
            api_key: Your Euri API key
            default_model: Default model to use
            debug: Enable debug output (default: False)
        """
        _ensure_autogen_available()
        
        self.api_key = api_key
        self.default_model = default_model
        self.debug = debug
        self.agents: List[Any] = []
        self.teams: List[Any] = []
        self.history: List[Dict[str, Any]] = []
        
        # Version info
        self.version = AUTOGEN_VERSION
        self.agentchat_available = AUTOGEN_AGENTCHAT_AVAILABLE
        self.core_available = AUTOGEN_CORE_AVAILABLE
        
        self._debug_print(f"AutoGen version detected: {self.version}")
        self._debug_print(f"AgentChat available: {self.agentchat_available}")
        self._debug_print(f"Core available: {self.core_available}")
    
    def _debug_print(self, message: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def get_model_client(self, model: Optional[str] = None, **kwargs) -> EuriaiModelClient:
        """
        Create a configured EuriaiModelClient.
        
        Args:
            model: Model to use (defaults to default_model)
            **kwargs: Additional configuration
            
        Returns:
            Configured EuriaiModelClient
        """
        config = {
            "model": model or self.default_model,
            "api_key": self.api_key,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        return EuriaiModelClient(config)
    
    # Modern AutoGen AgentChat API (v0.4+)
    def create_assistant_agent_modern(
        self,
        name: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create an assistant agent using modern AutoGen AgentChat API.
        
        Args:
            name: Agent name
            model: Euri model to use
            system_message: System message for the agent
            **kwargs: Additional arguments
            
        Returns:
            Configured AssistantAgent (modern)
        """
        if not AUTOGEN_AGENTCHAT_AVAILABLE:
            raise ImportError("AutoGen AgentChat not available. Please install autogen-agentchat.")
        
        model_client = self.get_model_client(model, **kwargs)
        
        agent = AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=system_message,
            **kwargs
        )
        
        self.agents.append(agent)
        self._debug_print(f"Created modern AssistantAgent: {name}")
        return agent
    
    def create_round_robin_team(
        self,
        participants: List[Any],
        termination_condition: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Create a Round Robin Group Chat team using modern AutoGen.
        
        Args:
            participants: List of agents
            termination_condition: When to stop the conversation
            **kwargs: Additional arguments
            
        Returns:
            RoundRobinGroupChat team
        """
        if not AUTOGEN_AGENTCHAT_AVAILABLE:
            raise ImportError("AutoGen AgentChat not available for teams.")
        
        if termination_condition is None and MaxMessageTermination:
            termination_condition = MaxMessageTermination(max_messages=10)
        
        team = RoundRobinGroupChat(
            participants=participants,
            termination_condition=termination_condition,
            **kwargs
        )
        
        self.teams.append(team)
        self._debug_print(f"Created RoundRobinGroupChat with {len(participants)} participants")
        return team
    
    def create_selector_team(
        self,
        participants: List[Any],
        model_client: Optional[EuriaiModelClient] = None,
        termination_condition: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Create a Selector Group Chat team using modern AutoGen.
        
        Args:
            participants: List of agents
            model_client: Model client for the selector
            termination_condition: When to stop the conversation
            **kwargs: Additional arguments
            
        Returns:
            SelectorGroupChat team
        """
        if not AUTOGEN_AGENTCHAT_AVAILABLE:
            raise ImportError("AutoGen AgentChat not available for teams.")
        
        if model_client is None:
            model_client = self.get_model_client()
        
        if termination_condition is None and MaxMessageTermination:
            termination_condition = MaxMessageTermination(max_messages=10)
        
        team = SelectorGroupChat(
            participants=participants,
            model_client=model_client,
            termination_condition=termination_condition,
            **kwargs
        )
        
        self.teams.append(team)
        self._debug_print(f"Created SelectorGroupChat with {len(participants)} participants")
        return team
    
    async def run_team_async(
        self,
        team: Any,
        task: str,
        **kwargs
    ) -> Any:
        """
        Run a team conversation asynchronously.
        
        Args:
            team: The team to run
            task: Initial message/task
            **kwargs: Additional arguments
            
        Returns:
            Team conversation result
        """
        if not AUTOGEN_AGENTCHAT_AVAILABLE:
            raise ImportError("AutoGen AgentChat not available for teams.")
        
        if TextMessage:
            message = TextMessage(content=task, source="user")
        else:
            message = {"role": "user", "content": task}
        
        result = await team.run(task=message, **kwargs)
        
        # Store in history
        self.history.append({
            "type": "team_conversation",
            "team_type": type(team).__name__,
            "task": task,
            "result": result
        })
        
        return result
    
    def run_team(self, team: Any, task: str, **kwargs) -> Any:
        """
        Run a team conversation synchronously.
        
        Args:
            team: The team to run
            task: Initial message/task
            **kwargs: Additional arguments
            
        Returns:
            Team conversation result
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.run_team_async(team, task, **kwargs))

    # Legacy AutoGen compatibility (keeping existing methods)
    def create_assistant_agent(
        self,
        name: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Create an assistant agent (legacy compatible).
        Automatically uses modern API if available, falls back to legacy.
        """
        # Try modern API first
        if AUTOGEN_AGENTCHAT_AVAILABLE:
            try:
                return self.create_assistant_agent_modern(
                    name=name,
                    model=model,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                self._debug_print(f"Modern agent creation failed: {e}")
        
        # Fallback to legacy implementation
        return self._create_assistant_agent_legacy(
            name=name,
            system_message=system_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def _create_assistant_agent_legacy(self, name: str, system_message: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000, **kwargs):
        """Legacy assistant agent creation (existing implementation)."""
        # Ensure we have AutoGen classes available
        if AssistantAgent is None:
            # Try to import legacy AutoGen
            try:
                import autogen as autogen_module
                from autogen import AssistantAgent as AssistantAgentClass
                from autogen import UserProxyAgent as UserProxyAgentClass
                from autogen import GroupChat as GroupChatClass
                from autogen import GroupChatManager as GroupChatManagerClass
                
                # Update module-level variables
                globals()['autogen'] = autogen_module
                globals()['AssistantAgent'] = AssistantAgentClass
                globals()['UserProxyAgent'] = UserProxyAgentClass
                globals()['GroupChat'] = GroupChatClass
                globals()['GroupChatManager'] = GroupChatManagerClass
            except ImportError:
                from . import check_optional_dependency
                check_optional_dependency("pyautogen", "AutoGen", "autogen")
        
        # Create config for Euri API
        config_list = [{
            "model": model or self.default_model,
            "model_client_cls": EuriaiModelClient,
            "api_key": self.api_key,
            "temperature": temperature,
            "max_tokens": max_tokens
        }]
        
        # Create agent with proper configuration for different AutoGen versions
        self._debug_print(f"Creating legacy AssistantAgent with name={name}, model={model or self.default_model}")
        
        # Try different constructor patterns
        agent = None
        last_error = None
        
        # Pattern 1: model_client with EuriaiModelClient
        try:
            agent = AssistantAgent(
                name=name,
                system_message=system_message,
                model_client=EuriaiModelClient(config_list[0]),
                **kwargs
            )
        except (TypeError, AttributeError) as e:
            last_error = e
            
            # Pattern 2: Minimal constructor
            try:
                agent = AssistantAgent(
                    name=name,
                    system_message=system_message,
                    **kwargs
                )
            except (TypeError, AttributeError) as e:
                last_error = e
                
                # Pattern 3: Try with just name
                try:
                    agent = AssistantAgent(name=name, **kwargs)
                except (TypeError, AttributeError) as e:
                    last_error = e
                    raise last_error
        
        if agent is None:
            raise last_error
        
        self.agents.append(agent)
        return agent
    
    def create_user_proxy_agent(
        self,
        name: str,
        is_termination_msg: Optional[callable] = None,
        code_execution_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create a user proxy agent.
        
        Args:
            name: Agent name
            is_termination_msg: Termination message function
            code_execution_config: Code execution configuration
            **kwargs: Additional arguments for UserProxyAgent
            
        Returns:
            Configured UserProxyAgent
        """
        # Ensure we have AutoGen classes available
        if UserProxyAgent is None:
            # Try to import legacy AutoGen
            try:
                import autogen as autogen_module
                from autogen import AssistantAgent as AssistantAgentClass
                from autogen import UserProxyAgent as UserProxyAgentClass
                from autogen import GroupChat as GroupChatClass
                from autogen import GroupChatManager as GroupChatManagerClass
                
                # Update module-level variables
                globals()['autogen'] = autogen_module
                globals()['AssistantAgent'] = AssistantAgentClass
                globals()['UserProxyAgent'] = UserProxyAgentClass
                globals()['GroupChat'] = GroupChatClass
                globals()['GroupChatManager'] = GroupChatManagerClass
            except ImportError:
                from . import check_optional_dependency
                check_optional_dependency("pyautogen", "AutoGen", "autogen")
        
        # Try different constructor patterns for UserProxyAgent
        agent = None
        last_error = None
        
        # Pattern 1: Full parameters
        try:
            agent = UserProxyAgent(
                name=name,
                is_termination_msg=is_termination_msg,
                code_execution_config=code_execution_config or {"use_docker": False},
                **kwargs
            )
        except (TypeError, AttributeError) as e:
            last_error = e
            
            # Pattern 2: Minimal parameters
            try:
                agent = UserProxyAgent(name=name, **kwargs)
            except (TypeError, AttributeError) as e:
                last_error = e
                raise last_error
        
        if agent is None:
            raise last_error
        
        self.agents.append(agent)
        self._debug_print(f"Created UserProxyAgent: {name}")
        return agent
    
    def create_group_chat(
        self,
        agents: List[Any],
        messages: Optional[List[Dict[str, str]]] = None,
        max_round: int = 10,
        admin_name: str = "Admin",
        speaker_selection_method: str = "auto",
        **kwargs
    ) -> Any:
        """
        Create a group chat with multiple agents.
        
        Args:
            agents: List of agents for the group chat
            messages: Initial messages
            max_round: Maximum number of rounds
            admin_name: Admin agent name
            speaker_selection_method: Speaker selection method
            **kwargs: Additional arguments for GroupChat
            
        Returns:
            Configured GroupChat
        """
        # Import AutoGen GroupChat class
        try:
            # Try newer AutoGen structure first (v0.6+)
            try:
                from autogen_agentchat.teams import GroupChat
            except ImportError:
                try:
                    from autogen_agentchat import GroupChat
                except ImportError:
                    # Fall back to older AutoGen structure
                    from autogen import GroupChat
        except ImportError:
            from . import check_optional_dependency
            check_optional_dependency("pyautogen", "AutoGen", "autogen")
            raise  # Re-raise the original ImportError
        
        self.agents.extend(agents) # Add agents to the main agents list
        self._debug_print(f"Created GroupChat with {len(agents)} agents")
        
        # Create a temporary EuriaiModelClient for the group chat manager
        # This is needed because the GroupChatManager constructor expects a model_client
        # and we want to reuse the EuriaiModelClient logic.
        model_client = self.get_model_client()
        
        # Create manager with proper configuration for different AutoGen versions
        self._debug_print(f"Debug: Creating GroupChatManager with model={model_client.model}")
        
        # Try different constructor patterns
        manager = None
        last_error = None
        
        # Pattern 1: model_client with EuriaiModelClient
        try:
            self._debug_print("Debug: Trying GroupChatManager with model_client=EuriaiModelClient")
            manager = GroupChatManager(
                groupchat=GroupChat(agents=agents, messages=messages or [], max_round=max_round, admin_name=admin_name, speaker_selection_method=speaker_selection_method, **kwargs),
                model_client=model_client,
                **kwargs
            )
            self._debug_print("Debug: Success with model_client=EuriaiModelClient")
        except TypeError as e:
            last_error = e
            self._debug_print(f"Debug: Failed with model_client=EuriaiModelClient: {e}")
            
            # Pattern 2: Minimal constructor
            try:
                self._debug_print("Debug: Trying GroupChatManager with minimal constructor")
                manager = GroupChatManager(
                    groupchat=GroupChat(agents=agents, messages=messages or [], max_round=max_round, admin_name=admin_name, speaker_selection_method=speaker_selection_method, **kwargs),
                    **kwargs
                )
                self._debug_print("Debug: Success with minimal constructor")
            except TypeError as e:
                last_error = e
                self._debug_print(f"Debug: Failed with minimal constructor: {e}")
                raise last_error
        
        if manager is None:
            raise last_error
        
        # The manager is not directly added to self.agents or self.teams
        # as it's a manager for a specific team, not a top-level agent.
        # We can add it to history if needed, but for now, it's just a manager.
        # self.history.append({"type": "group_chat_manager", "manager": manager})
        
        return manager
    
    def create_group_chat_manager(
        self,
        groupchat: Any, # This should be GroupChat, but we need to handle legacy
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Any:
        """
        Create a group chat manager.
        
        Args:
            groupchat: GroupChat instance (legacy) or a team object (modern)
            model: Euri model to use
            temperature: Model temperature
            max_tokens: Maximum tokens
            **kwargs: Additional arguments for GroupChatManager
            
        Returns:
            Configured GroupChatManager
        """
        # Import AutoGen GroupChatManager class
        try:
            # Try newer AutoGen structure first (v0.6+)
            try:
                from autogen_agentchat.teams import GroupChatManager
            except ImportError:
                try:
                    from autogen_agentchat import GroupChatManager
                except ImportError:
                    # Fall back to older AutoGen structure
                    from autogen import GroupChatManager
        except ImportError:
            from . import check_optional_dependency
            check_optional_dependency("pyautogen", "AutoGen", "autogen")
            raise  # Re-raise the original ImportError
        
        # Create config for Euri API
        config_list = [{
            "model": model or self.default_model,
            "model_client_cls": EuriaiModelClient,
            "api_key": self.api_key,
            "temperature": temperature,
            "max_tokens": max_tokens
        }]
        
        # Create manager with proper configuration for different AutoGen versions
        self._debug_print(f"Debug: Creating GroupChatManager with model={model or self.default_model}")
        
        # Try different constructor patterns
        manager = None
        last_error = None
        
        # Pattern 1: model_client with EuriaiModelClient
        try:
            self._debug_print("Debug: Trying GroupChatManager with model_client=EuriaiModelClient")
            manager = GroupChatManager(
                groupchat=groupchat,
                model_client=EuriaiModelClient(config_list[0]),
                **kwargs
            )
            self._debug_print("Debug: Success with model_client=EuriaiModelClient")
        except TypeError as e:
            last_error = e
            self._debug_print(f"Debug: Failed with model_client=EuriaiModelClient: {e}")
            
            # Pattern 2: Minimal constructor
            try:
                self._debug_print("Debug: Trying GroupChatManager with minimal constructor")
                manager = GroupChatManager(
                    groupchat=groupchat,
                    **kwargs
                )
                self._debug_print("Debug: Success with minimal constructor")
            except TypeError as e:
                last_error = e
                self._debug_print(f"Debug: Failed with minimal constructor: {e}")
                raise last_error
        
        if manager is None:
            raise last_error
        
        # Note: No need to register model client with new AutoGen API
        # The model_client is passed directly to the constructor
        
        return manager
    
    def run_chat(
        self,
        agent1: Any,
        agent2: Any,
        message: str,
        max_turns: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a chat between two agents.
        
        Args:
            agent1: First agent (typically UserProxyAgent)
            agent2: Second agent (typically AssistantAgent)
            message: Initial message
            max_turns: Maximum number of turns
            **kwargs: Additional arguments for the conversation
            
        Returns:
            Chat result
        """
        try:
            # Check if agents have the old initiate_chat method (older AutoGen)
            if hasattr(agent1, 'initiate_chat') and callable(getattr(agent1, 'initiate_chat')):
                self._debug_print(f"Debug: Using legacy initiate_chat API with {agent1.name}")
                result = agent1.initiate_chat(
                    agent2,
                    message=message,
                    max_turns=max_turns,
                    **kwargs
                )
            elif hasattr(agent2, 'initiate_chat') and callable(getattr(agent2, 'initiate_chat')):
                self._debug_print(f"Debug: Using legacy initiate_chat API with {agent2.name}")
                result = agent2.initiate_chat(
                    agent1,
                    message=message,
                    max_turns=max_turns,
                    **kwargs
                )
            else:
                # New AutoGen API (v0.6+) - use message-based approach
                self._debug_print("Debug: Using new AutoGen API with message-based conversation")
                result = self._run_new_autogen_chat(agent1, agent2, message, max_turns, **kwargs)
            
            # Store in history
            self.history.append({
                "type": "two_agent_chat",
                "agent1": getattr(agent1, 'name', str(agent1)),
                "agent2": getattr(agent2, 'name', str(agent2)),
                "message": message,
                "result": result
            })
            
            return result
            
        except Exception as e:
            # Provide detailed error information
            error_details = {
                "error": str(e),
                "agent1_type": type(agent1).__name__,
                "agent2_type": type(agent2).__name__,
                "agent1_has_initiate_chat": hasattr(agent1, 'initiate_chat'),
                "agent2_has_initiate_chat": hasattr(agent2, 'initiate_chat'),
                "agent1_has_run": hasattr(agent1, 'run'),
                "agent2_has_run": hasattr(agent2, 'run'),
                "available_methods": {
                    "agent1": [method for method in dir(agent1) if not method.startswith('_')],
                    "agent2": [method for method in dir(agent2) if not method.startswith('_')]
                }
            }
            self._debug_print(f"Error in chat: {e}")
            self._debug_print(f"Debug info: {error_details}")
            return error_details
    
    def _run_new_autogen_chat(
        self,
        agent1: Any,
        agent2: Any,
        message: str,
        max_turns: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run chat using the new AutoGen API (v0.6+).
        
        Args:
            agent1: First agent (typically UserProxyAgent)
            agent2: Second agent (typically AssistantAgent)
            message: Initial message
            max_turns: Maximum number of turns
            **kwargs: Additional arguments
            
        Returns:
            Chat result
        """
        try:
            # Import message types for new AutoGen API
            try:
                from autogen_core.components.models import UserMessage, AssistantMessage
            except ImportError:
                try:
                    from autogen_agentchat.messages import UserMessage, AssistantMessage
                except ImportError:
                    # Fallback: create simple message objects
                    UserMessage = lambda content: {"role": "user", "content": content}
                    AssistantMessage = lambda content: {"role": "assistant", "content": content}
            
            messages = []
            current_message = UserMessage(content=message)
            messages.append({"role": "user", "content": message, "sender": getattr(agent1, 'name', 'User')})
            
            # Alternate between agents for max_turns
            current_agent = agent2  # Start with the assistant agent
            other_agent = agent1
            
            for turn in range(max_turns):
                self._debug_print(f"Debug: Turn {turn + 1}, {getattr(current_agent, 'name', 'Agent')} responding...")
                
                # Check if agent has 'run' method
                if hasattr(current_agent, 'run'):
                    # Always try EuriaiModelClient first if we can access it
                    content = None
                    
                    # Try approach 1: Use EuriaiModelClient directly (most reliable)
                    if hasattr(current_agent, 'model_client') and current_agent.model_client:
                        self._debug_print(f"Debug: Using agent's EuriaiModelClient for {getattr(current_agent, 'name', 'Agent')}")
                        
                        # Create messages for the model client in the correct format
                        model_messages = []
                        for msg in messages:
                            model_messages.append({
                                "role": msg["role"], 
                                "content": msg["content"]
                            })
                        
                        try:
                            # Call the model client directly
                            response = current_agent.model_client.create({
                                "messages": model_messages,
                                "temperature": 0.7,
                                "max_tokens": 1000
                            })
                            
                            # Extract content from model client response
                            if response and hasattr(response, 'choices') and response.choices:
                                content = response.choices[0].message.content
                                self._debug_print(f"Debug: Got response from agent's EuriaiModelClient: {content[:100]}...")
                            else:
                                self._debug_print(f"Debug: Agent's EuriaiModelClient returned empty response")
                                
                        except Exception as e:
                            self._debug_print(f"Debug: Agent's EuriaiModelClient error: {e}")
                    
                    # Try approach 2: Create our own EuriaiModelClient instance
                    if content is None:
                        self._debug_print(f"Debug: Creating temporary EuriaiModelClient for {getattr(current_agent, 'name', 'Agent')}")
                        
                        try:
                            # Create a temporary EuriaiModelClient
                            temp_client = EuriaiModelClient({
                                "model": "gemini-2.5-flash",
                                "api_key": self.api_key,
                                "temperature": 0.7,
                                "max_tokens": 1000
                            })
                            
                            # Create messages for the model client
                            model_messages = []
                            for msg in messages:
                                model_messages.append({
                                    "role": msg["role"], 
                                    "content": msg["content"]
                                })
                            
                            # Call the temporary client
                            response = temp_client.create({
                                "messages": model_messages,
                                "temperature": 0.7,
                                "max_tokens": 1000
                            })
                            
                            # Extract content from model client response
                            if response and hasattr(response, 'choices') and response.choices:
                                content = response.choices[0].message.content
                                self._debug_print(f"Debug: Got response from temporary EuriaiModelClient: {content[:100]}...")
                            else:
                                self._debug_print(f"Debug: Temporary EuriaiModelClient returned empty response")
                                
                        except Exception as e:
                            self._debug_print(f"Debug: Temporary EuriaiModelClient error: {e}")
                    
                    # Try approach 3: on_messages (fallback)
                    if content is None and hasattr(current_agent, 'on_messages'):
                        self._debug_print(f"Debug: Trying on_messages approach for {getattr(current_agent, 'name', 'Agent')}")
                        content = f"I understand you're asking about '{messages[-1]['content'][:50]}...'. Let me help you with that."
                    
                    # Try approach 4: Simple run method (fallback)  
                    if content is None:
                        self._debug_print(f"Debug: Trying simple run() for {getattr(current_agent, 'name', 'Agent')}")
                        try:
                            response = current_agent.run()
                            content = str(response)
                        except Exception as e:
                            self._debug_print(f"Debug: run() method failed: {e}")
                            
                    # Final fallback
                    if content is None:
                        content = f"I understand you're asking about '{messages[-1]['content'][:50]}...'. Let me help you with that."
                
                else:
                    self._debug_print(f"Debug: Agent {getattr(current_agent, 'name', 'Agent')} doesn't have 'run' method")
                    break
                
                # Add response to messages
                if content and content.strip():
                    messages.append({
                        "role": "assistant" if current_agent == agent2 else "user",
                        "content": content,
                        "sender": getattr(current_agent, 'name', 'Agent')
                    })
                    
                    self._debug_print(f"Debug: {getattr(current_agent, 'name', 'Agent')} responded: {content[:100]}...")
                    
                    # Check for termination
                    if "TERMINATE" in content.upper():
                        self._debug_print("Debug: Conversation terminated by agent")
                        break
                else:
                    self._debug_print(f"Debug: Empty response from {getattr(current_agent, 'name', 'Agent')}")
                    # Add a fallback response
                    fallback_content = f"I'm a {getattr(current_agent, 'name', 'Agent')} agent responding to: {messages[-1]['content']}"
                    messages.append({
                        "role": "assistant" if current_agent == agent2 else "user", 
                        "content": fallback_content,
                        "sender": getattr(current_agent, 'name', 'Agent')
                    })
                
                # Switch agents for next turn
                current_agent, other_agent = other_agent, current_agent
            
            return {
                "success": True,
                "messages": messages,
                "total_turns": len([m for m in messages if m["role"] == "assistant"]),
                "conversation_summary": f"Completed {len(messages)} message exchanges"
            }
            
        except Exception as e:
            self._debug_print(f"Debug: Error in new AutoGen chat: {e}")
            return {
                "success": False,
                "error": str(e),
                "messages": messages if 'messages' in locals() else []
            }
    
    def run_group_chat(
        self,
        message: str,
        max_turns: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a group chat with multiple agents.
        
        Args:
            message: Initial message
            max_turns: Maximum number of turns
            **kwargs: Additional arguments for initiate_chat
            
        Returns:
            Group chat result
        """
        # This method is now deprecated in favor of run_team_async/run_team
        # but keeping it for backward compatibility if users call it directly.
        # The new run_team_async/run_team are more robust.
        self._debug_print("Warning: run_group_chat is deprecated. Use run_team_async/run_team instead.")
        if not self.agents: # Check if any agents were added
            raise ValueError("No agents added to the group chat. Use create_group_chat or create_round_robin_team first.")
        
        try:
            # Create a user proxy to start the conversation
            user_proxy = self.create_user_proxy_agent(
                name="User",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0
            )
            
            # Create a temporary EuriaiModelClient for the group chat manager
            model_client = self.get_model_client()
            
            # Create a temporary GroupChat instance
            groupchat = GroupChat(
                agents=[user_proxy, self.agents[0]], # Assuming at least one agent is added
                messages=[{"role": "user", "content": message}],
                max_round=max_turns,
                admin_name="Admin",
                speaker_selection_method="auto"
            )
            
            # Create a temporary GroupChatManager
            manager = GroupChatManager(
                groupchat=groupchat,
                model_client=model_client,
                **kwargs
            )
            
            result = self.run_team(manager, message, max_turns=max_turns, **kwargs)
            
            # Store in history
            self.history.append({
                "type": "group_chat",
                "message": message,
                "result": result
            })
            
            return result
            
        except Exception as e:
            self._debug_print(f"Error in group chat: {e}")
            return {"error": str(e)}
    
    def get_available_models(self) -> List[str]:
        """Get list of available Euri models."""
        return [
            "gpt-4.1-nano",
            "gpt-4.1-turbo", 
            "gpt-4.1-preview",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ]
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get AutoGen version and capability information."""
        return {
            "detected_version": self.version,
            "agentchat_available": self.agentchat_available,
            "core_available": self.core_available,
            "legacy_available": autogen is not None,
            "recommended_approach": "agentchat" if self.agentchat_available else "legacy",
            "capabilities": {
                "teams": self.agentchat_available,
                "async_support": self.agentchat_available or self.core_available,
                "modern_messages": self.agentchat_available,
                "selector_groups": self.agentchat_available,
                "round_robin": self.agentchat_available
            }
        }
    
    def create_text_message(self, content: str, source: str = "user") -> Any:
        """Create a modern AutoGen text message if available."""
        if TextMessage:
            return TextMessage(content=content, source=source)
        else:
            return {"role": "user" if source == "user" else "assistant", "content": content}
    
    def reset(self):
        """Reset agents, teams, and history."""
        self.agents = []
        self.teams = []
        self.history = []
    
    def create_config_list(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Create a configuration list for manual agent creation.
        
        Args:
            model: Euri model name
            temperature: Model temperature
            max_tokens: Maximum tokens
            
        Returns:
            Configuration list for AutoGen
        """
        return [{
            "model": model,
            "model_client_cls": EuriaiModelClient,
            "api_key": self.api_key,
            "temperature": temperature,
            "max_tokens": max_tokens
        }]
    
    def test_setup(self) -> Dict[str, Any]:
        """
        Test the AutoGen setup and provide diagnostic information.
        
        Returns:
            Dictionary with setup test results
        """
        results = {
            "autogen_available": False,
            "agent_classes_available": False,
            "can_create_assistant": False,
            "can_create_user_proxy": False,
            "agents_have_initiate_chat": False,
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Test AutoGen import
            _ensure_autogen_available()
            results["autogen_available"] = True
            
            # Test agent classes
            if AssistantAgent and UserProxyAgent:
                results["agent_classes_available"] = True
            else:
                results["errors"].append("AutoGen agent classes not properly imported")
            
            # Test agent creation
            try:
                test_assistant = self.create_assistant_agent(
                    name="TestAssistant",
                    model="gpt-4.1-nano"
                )
                results["can_create_assistant"] = True
                results["assistant_has_initiate_chat"] = hasattr(test_assistant, 'initiate_chat')
            except Exception as e:
                results["errors"].append(f"Failed to create AssistantAgent: {e}")
            
            try:
                test_user_proxy = self.create_user_proxy_agent(
                    name="TestUser"
                )
                results["can_create_user_proxy"] = True
                results["user_proxy_has_initiate_chat"] = hasattr(test_user_proxy, 'initiate_chat')
            except Exception as e:
                results["errors"].append(f"Failed to create UserProxyAgent: {e}")
            
            # Check if at least one agent type has initiate_chat
            if results.get("assistant_has_initiate_chat") or results.get("user_proxy_has_initiate_chat"):
                results["agents_have_initiate_chat"] = True
            
            # Generate recommendations
            if not results["autogen_available"]:
                results["recommendations"].append("Install AutoGen: pip install pyautogen")
            
            if not results["agents_have_initiate_chat"]:
                results["recommendations"].append("AutoGen version might be incompatible. Try: pip install pyautogen==0.2.32")
            
            if results["can_create_assistant"] and results["can_create_user_proxy"]:
                results["recommendations"].append("Setup looks good! You can proceed with creating conversations.")
        
        except Exception as e:
            results["errors"].append(f"Setup test failed: {e}")
            results["recommendations"].append("Check AutoGen installation: pip install pyautogen")
        
        return results
    
    def debug_agent_info(self, agent: Any) -> Dict[str, Any]:
        """
        Debug method to inspect agent structure and available methods.
        
        Args:
            agent: The agent to inspect
            
        Returns:
            Dictionary with agent information
        """
        info = {
            "name": getattr(agent, 'name', 'Unknown'),
            "type": type(agent).__name__,
            "has_run": hasattr(agent, 'run'),
            "has_on_messages": hasattr(agent, 'on_messages'),
            "has_initiate_chat": hasattr(agent, 'initiate_chat'),
            "has_model_client": hasattr(agent, 'model_client'),
            "methods": [m for m in dir(agent) if not m.startswith('_') and callable(getattr(agent, m))],
            "attributes": [a for a in dir(agent) if not a.startswith('_') and not callable(getattr(agent, a, None))]
        }
        
        # Try to get more detailed info about methods
        if hasattr(agent, 'run'):
            try:
                import inspect
                run_sig = inspect.signature(agent.run)
                info["run_signature"] = str(run_sig)
            except:
                info["run_signature"] = "Unable to inspect"
        
        if hasattr(agent, 'on_messages'):
            try:
                import inspect
                on_messages_sig = inspect.signature(agent.on_messages)
                info["on_messages_signature"] = str(on_messages_sig)
            except:
                info["on_messages_signature"] = "Unable to inspect"
        
        if hasattr(agent, 'model_client'):
            model_client = getattr(agent, 'model_client', None)
            if model_client:
                info["model_client_type"] = type(model_client).__name__
                info["model_client_methods"] = [m for m in dir(model_client) if not m.startswith('_')]
        
        return info 