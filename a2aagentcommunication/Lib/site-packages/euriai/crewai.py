import os
from typing import Optional, Dict, Any, List, Union
from euriai.client import EuriaiClient

# CrewAI imports (user must install crewai)
try:
    from crewai import Agent, Crew, Task, Process
    from crewai.llm import LLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Fallback base classes for when CrewAI is not available
    class Agent:
        pass
    class Crew:
        pass
    class Task:
        pass
    class Process:
        sequential = "sequential"
        parallel = "parallel"
    class LLM:
        pass

class EuriaiLLM(LLM):
    """Custom LLM that uses Euri API for CrewAI agents"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano", temperature: float = 0.7, max_tokens: int = 1000):
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. Please install with: "
                "pip install crewai"
            )
        
        self.client = EuriaiClient(api_key=api_key, model=model)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def call(self, prompt: str, **kwargs) -> str:
        """Make a call to the Euri API"""
        try:
            response = self.client.generate_completion(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            return f"Error calling Euri API: {str(e)}"

class EuriaiCrewAI:
    """
    Enhanced CrewAI integration that uses Euri API for LLM calls.
    """
    
    def __init__(
        self, 
        api_key: str,
        default_model: str = "gpt-4.1-nano",
        agents: Optional[Dict[str, Any]] = None, 
        tasks: Optional[Dict[str, Any]] = None, 
        process: str = "sequential", 
        verbose: bool = True
    ):
        """
        Initialize the CrewAI wrapper with Euri API integration.
        
        Args:
            api_key: Your Euri API key
            default_model: Default model to use (e.g., 'gpt-4.1-nano', 'claude-3-5-sonnet')
            agents: Dict of agent configs
            tasks: Dict of task configs  
            process: 'sequential' or 'parallel'
            verbose: Print detailed logs
        """
        if Agent is None:
            raise ImportError("CrewAI is not installed. Please install with `pip install crewai`.")
        
        self.api_key = api_key
        self.default_model = default_model
        self.agents_config = agents or {}
        self.tasks_config = tasks or {}
        self.process = Process.sequential if process == "sequential" else Process.parallel
        self.verbose = verbose
        self._agents: List[Agent] = []
        self._tasks: List[Task] = []
        self._crew: Optional[Crew] = None

    def _create_euri_llm(self, model: str = None, temperature: float = 0.7, max_tokens: int = 1000) -> EuriaiLLM:
        """Create an EuriaiLLM instance"""
        return EuriaiLLM(
            api_key=self.api_key,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def add_agent(self, name: str, config: Dict[str, Any]) -> None:
        """
        Add an agent with Euri API integration.
        
        Args:
            name: Agent name
            config: Agent configuration. Can include:
                - model: Euri model to use (e.g., 'gpt-4.1-nano', 'claude-3-5-sonnet')
                - temperature: Model temperature
                - max_tokens: Max tokens for responses
                - role: Agent role
                - goal: Agent goal
                - backstory: Agent backstory
        """
        # Extract LLM config
        model = config.pop('model', self.default_model)
        temperature = config.pop('temperature', 0.7)
        max_tokens = config.pop('max_tokens', 1000)
        
        # Create Euri LLM instance
        euri_llm = self._create_euri_llm(model, temperature, max_tokens)
        
        # Add LLM to agent config
        config['llm'] = euri_llm
        
        # Create agent
        agent = Agent(**config)
        self._agents.append(agent)
        self.agents_config[name] = config

    def add_task(self, name: str, config: Dict[str, Any]) -> None:
        """Add a task by config."""
        # Handle agent reference
        if 'agent' in config and isinstance(config['agent'], str):
            # Find agent by name
            agent_name = config['agent']
            for i, agent_config in enumerate(self.agents_config.values()):
                if i == list(self.agents_config.keys()).index(agent_name):
                    config['agent'] = self._agents[i]
                    break
        
        task = Task(**config)
        self._tasks.append(task)
        self.tasks_config[name] = config

    def build_crew(self) -> Crew:
        """Build the Crew object from current agents and tasks."""
        if not self._agents:
            for name, cfg in self.agents_config.items():
                self.add_agent(name, cfg.copy())
        if not self._tasks:
            for name, cfg in self.tasks_config.items():
                self.add_task(name, cfg.copy())
        
        self._crew = Crew(
            agents=self._agents, 
            tasks=self._tasks, 
            process=self.process, 
            verbose=self.verbose
        )
        return self._crew

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run the crew workflow with Euri API integration.
        Returns the final result or report.
        """
        if self._crew is None:
            self.build_crew()
        return self._crew.kickoff(inputs=inputs or {})

    @classmethod
    def from_yaml(
        cls, 
        agents_yaml: str, 
        tasks_yaml: str, 
        api_key: str,
        default_model: str = "gpt-4.1-nano",
        process: str = "sequential", 
        verbose: bool = True
    ):
        """
        Create a CrewAI wrapper from YAML config files with Euri API integration.
        
        Args:
            agents_yaml: Path to agents.yaml
            tasks_yaml: Path to tasks.yaml
            api_key: Your Euri API key
            default_model: Default model to use
        """
        import yaml
        with open(agents_yaml, "r") as f:
            agents = yaml.safe_load(f)
        with open(tasks_yaml, "r") as f:
            tasks = yaml.safe_load(f)
        return cls(
            api_key=api_key,
            default_model=default_model,
            agents=agents, 
            tasks=tasks, 
            process=process, 
            verbose=verbose
        )

    def get_agents(self) -> List[Agent]:
        return self._agents

    def get_tasks(self) -> List[Task]:
        return self._tasks

    def get_crew(self) -> Optional[Crew]:
        return self._crew

    def reset(self):
        """Reset agents, tasks, and crew."""
        self._agents = []
        self._tasks = []
        self._crew = None

    def list_available_models(self) -> List[str]:
        """List available Euri models"""
        return [
            "gpt-4.1-nano",
            "gpt-4.1-mini", 
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-pro-preview-06-05",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-flash-lite-preview-06-17",
            "gemma2-9b-it"
        ] 