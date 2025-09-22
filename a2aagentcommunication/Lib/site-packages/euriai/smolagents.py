import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
import uuid

try:
    from smolagents import CodeAgent, ApiModel, tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    CodeAgent = ApiModel = tool = None
    SMOLAGENTS_AVAILABLE = False

from .client import EuriaiClient


class AgentType(Enum):
    """Pre-defined agent types for different use cases."""
    CODER = "coder"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CREATIVE = "creative"
    ASSISTANT = "assistant"
    PROBLEM_SOLVER = "problem_solver"
    MULTI_TOOL = "multi_tool"
    SPECIALIST = "specialist"


@dataclass
class AgentConfig:
    """Configuration for agent behavior and capabilities."""
    name: str
    description: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_iterations: int = 10
    timeout: int = 300
    enable_memory: bool = True
    enable_streaming: bool = False
    error_recovery: bool = True
    log_level: str = "INFO"


@dataclass
class TaskResult:
    """Result of an agent task execution."""
    agent_id: str
    task_id: str
    task: str
    result: Any
    success: bool
    execution_time: float
    iterations: int
    error_message: Optional[str] = None
    usage_stats: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AgentMemory:
    """Memory management for agent interactions."""
    agent_id: str
    memories: List[Dict[str, Any]]
    max_memories: int = 100

    def add_memory(self, memory: Dict[str, Any]):
        """Add a memory entry."""
        self.memories.append(memory)
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)

    def get_recent_memories(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories."""
        return self.memories[-count:]

    def clear_memories(self):
        """Clear all memories."""
        self.memories.clear()


class EuriaiTool:
    """Enhanced tool wrapper for Euri API integration."""
    
    def __init__(self, client: EuriaiClient, name: str, description: str, 
                 function: Callable, model: str = "gpt-4o"):
        self.client = client
        self.name = name
        self.description = description
        self.function = function
        self.model = model
        self.usage_count = 0
        self.total_time = 0.0
        
    def __call__(self, *args, **kwargs):
        """Execute the tool with AI enhancement."""
        start_time = time.time()
        try:
            # Check if tool needs AI assistance
            if hasattr(self.function, '_needs_ai') and self.function._needs_ai:
                # Enhance tool execution with AI
                enhanced_result = self._enhance_with_ai(*args, **kwargs)
                result = enhanced_result
            else:
                result = self.function(*args, **kwargs)
            
            execution_time = time.time() - start_time
            self.usage_count += 1
            self.total_time += execution_time
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self.total_time += execution_time
            raise e
    
    def _enhance_with_ai(self, *args, **kwargs):
        """Enhance tool execution with AI assistance."""
        # Create AI prompt for tool enhancement
        prompt = f"""
        You are helping execute a tool called '{self.name}' with description: {self.description}
        
        Tool arguments: {args}
        Tool keyword arguments: {kwargs}
        
        Please provide intelligent assistance for this tool execution.
        """
        
        response = self.client.generate_completion(
            prompt=prompt,
            temperature=0.5
        )
        
        ai_result = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Combine AI result with tool function
        original_result = self.function(*args, **kwargs)
        
        return {
            "original_result": original_result,
            "ai_enhancement": ai_result,
            "combined_result": f"{original_result}\n\nAI Enhancement: {ai_result}"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "total_time": self.total_time,
            "average_time": self.total_time / max(self.usage_count, 1)
        }


class EuriaiSmolAgent:
    """Enhanced SmolAgents integration with Euri API."""
    
    def __init__(self, 
                 api_key: str,
                 config: Optional[AgentConfig] = None,
                 tools: Optional[List[Callable]] = None):
        """
        Initialize enhanced SmolAgent with Euri API integration.
        
        Args:
            api_key: Euri API key
            config: Agent configuration
            tools: List of tool functions
        """
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError("SmolAgents is not installed. Please install with `pip install smolagents`.")
        
        self.client = EuriaiClient(api_key=api_key)
        self.config = config or AgentConfig(
            name="Default Agent",
            description="A general-purpose AI agent"
        )
        self.agent_id = str(uuid.uuid4())
        
        # Initialize components
        self.tools = {}
        self.memory = AgentMemory(agent_id=self.agent_id, memories=[])
        self.task_history = []
        self.usage_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "total_time": 0.0,
            "api_calls": 0,
            "errors": 0
        }
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(f"EuriaiSmolAgent-{self.agent_id}")
        
        # Initialize SmolAgents
        self.model = self._create_euri_model()
        self.agent = CodeAgent(tools=list(self.tools.values()), model=self.model)
        
        # Add initial tools
        if tools:
            for tool_fn in tools:
                self.add_tool(tool_fn)
    
    def _create_euri_model(self):
        """Create a model interface for SmolAgents using Euri API."""
        
        # Define ModelResponse class outside the method for better error handling
        class ModelResponse:
            def __init__(self, content, usage=None):
                self.content = content
                self.text = content  # Alternative attribute name
                # Add token usage info for SmolAgents compatibility
                if usage is None:
                    usage = {
                        'input_tokens': 0,  # SmolAgents expected names
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'prompt_tokens': 0,  # Also include original names
                        'completion_tokens': 0
                    }
                
                # Create token usage object with attributes
                class TokenUsage:
                    def __init__(self, usage_dict):
                        self.input_tokens = usage_dict.get('input_tokens', 0)
                        self.output_tokens = usage_dict.get('output_tokens', 0)
                        self.total_tokens = usage_dict.get('total_tokens', 0)
                        self.prompt_tokens = usage_dict.get('prompt_tokens', 0)
                        self.completion_tokens = usage_dict.get('completion_tokens', 0)
                
                self.token_usage = TokenUsage(usage)
            
            def __str__(self):
                return self.content
        
        class EuriModel:
            def __init__(self, client, config):
                self.client = client
                self.config = config
            
            def __call__(self, messages, **kwargs):
                """Make API call using Euri client."""
                try:
                    # Convert messages to a single prompt
                    if isinstance(messages, list):
                        prompt_parts = []
                        for msg in messages:
                            if isinstance(msg, dict) and 'content' in msg:
                                role = msg.get('role', 'user')
                                content = msg['content']
                                if role == 'system':
                                    prompt_parts.append(f"System: {content}")
                                elif role == 'user':
                                    prompt_parts.append(f"User: {content}")
                                elif role == 'assistant':
                                    prompt_parts.append(f"Assistant: {content}")
                                else:
                                    prompt_parts.append(content)
                            else:
                                prompt_parts.append(str(msg))
                        prompt = "\n".join(prompt_parts)
                    else:
                        prompt = str(messages)
                    
                    # Filter kwargs to only include supported parameters
                    supported_params = {
                        'temperature', 'max_tokens', 'top_p', 
                        'frequency_penalty', 'presence_penalty', 'stop'
                    }
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
                    
                    # Map stop_sequences to stop if provided
                    if 'stop_sequences' in kwargs:
                        filtered_kwargs['stop'] = kwargs['stop_sequences']
                    
                    # Use EuriaiClient's generate_completion method
                    response = self.client.generate_completion(
                        prompt=prompt,
                        temperature=self.config.temperature,
                        **filtered_kwargs
                    )
                    
                    # Extract content from response
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Extract usage information if available
                    usage_info = response.get("usage", {})
                    token_usage = {
                        'input_tokens': usage_info.get('prompt_tokens', 0),  # Map to SmolAgents expected names
                        'output_tokens': usage_info.get('completion_tokens', 0),
                        'total_tokens': usage_info.get('total_tokens', 0),
                        # Also include original names for compatibility
                        'prompt_tokens': usage_info.get('prompt_tokens', 0),
                        'completion_tokens': usage_info.get('completion_tokens', 0)
                    }
                    
                    return ModelResponse(content, token_usage)
                    
                except Exception as e:
                    logging.error(f"Error in Euri API call: {e}")
                    # Return a proper ModelResponse object with a more graceful error message
                    if "403" in str(e) or "401" in str(e):
                        error_content = "I apologize, but I'm unable to access the AI service at the moment due to authentication issues. Please check your API key and try again."
                    elif "429" in str(e):
                        error_content = "I apologize, but the service is currently experiencing high demand. Please try again in a moment."
                    else:
                        error_content = f"I apologize, but I encountered an error while processing your request: {str(e)}"
                    return ModelResponse(error_content, None)
            
            def generate(self, prompt, **kwargs):
                """Generate method expected by SmolAgents."""
                # Handle both string prompts and message lists
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = prompt
                return self.__call__(messages, **kwargs)
        
        return EuriModel(self.client, self.config)
    
    def add_tool(self, tool_fn: Callable, name: Optional[str] = None, 
                description: Optional[str] = None, model: Optional[str] = None) -> None:
        """Add an enhanced tool to the agent."""
        tool_name = name or getattr(tool_fn, '__name__', 'unnamed_tool')
        tool_description = description or getattr(tool_fn, '__doc__', 'No description provided')
        tool_model = model or self.config.model
        
        enhanced_tool = EuriaiTool(
            client=self.client,
            name=tool_name,
            description=tool_description,
            function=tool_fn,
            model=tool_model
        )
        
        self.tools[tool_name] = enhanced_tool
        
        # Update SmolAgent with new tools
        self.agent = CodeAgent(tools=list(self.tools.values()), model=self.model)
        
        self.logger.info(f"Added tool: {tool_name}")
    
    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.agent = CodeAgent(tools=list(self.tools.values()), model=self.model)
            self.logger.info(f"Removed tool: {tool_name}")
    
    def run(self, task: str, **kwargs) -> TaskResult:
        """
        Run a task with enhanced features.
        
        Args:
            task: Task description
            **kwargs: Additional parameters
            
        Returns:
            TaskResult with execution details
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Starting task {task_id}: {task}")
        
        try:
            # Add task to memory
            if self.config.enable_memory:
                self.memory.add_memory({
                    "type": "task_start",
                    "task_id": task_id,
                    "task": task,
                    "timestamp": datetime.now()
                })
            
            # Execute with SmolAgents
            result = self.agent.run(task, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Check if the result contains an error message
            result_str = str(result)
            is_error = any(error_indicator in result_str.lower() for error_indicator in [
                'error:', '403', '401', '429', 'forbidden', 'unauthorized', 'rate limit'
            ])
            
            # Create result object
            task_result = TaskResult(
                agent_id=self.agent_id,
                task_id=task_id,
                task=task,
                result=result,
                success=not is_error,  # Mark as failed if error detected
                execution_time=execution_time,
                iterations=1,  # SmolAgents doesn't provide iteration count
                usage_stats=self._get_usage_stats(),
                error_message=result_str if is_error else None
            )
            
            # Update statistics
            self.usage_stats["total_tasks"] += 1
            if not is_error:
                self.usage_stats["successful_tasks"] += 1
            else:
                self.usage_stats["errors"] += 1
            self.usage_stats["total_time"] += execution_time
            self.usage_stats["api_calls"] += 1
            
            # Add to history
            self.task_history.append(task_result)
            
            # Add to memory
            if self.config.enable_memory:
                self.memory.add_memory({
                    "type": "task_complete",
                    "task_id": task_id,
                    "result": result,
                    "success": True,
                    "timestamp": datetime.now()
                })
            
            self.logger.info(f"Task {task_id} completed successfully")
            
            return task_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle errors
            error_message = str(e)
            
            if self.config.error_recovery:
                error_message = self._handle_error(e, task, task_id)
            
            task_result = TaskResult(
                agent_id=self.agent_id,
                task_id=task_id,
                task=task,
                result=None,
                success=False,
                execution_time=execution_time,
                iterations=1,
                error_message=error_message
            )
            
            # Update statistics
            self.usage_stats["total_tasks"] += 1
            self.usage_stats["total_time"] += execution_time
            
            # Add to history
            self.task_history.append(task_result)
            
            # Add to memory
            if self.config.enable_memory:
                self.memory.add_memory({
                    "type": "task_error",
                    "task_id": task_id,
                    "error": error_message,
                    "timestamp": datetime.now()
                })
            
            self.logger.error(f"Task {task_id} failed: {error_message}")
            
            return task_result
    
    async def run_async(self, task: str, **kwargs) -> TaskResult:
        """Run a task asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run, task, **kwargs
        )
    
    async def run_streaming(self, task: str, **kwargs) -> AsyncGenerator[str, None]:
        """Run a task with streaming output."""
        if not self.config.enable_streaming:
            raise ValueError("Streaming is not enabled for this agent")
        
        # Note: SmolAgents doesn't natively support streaming
        # This is a simulated streaming implementation
        task_id = str(uuid.uuid4())
        
        yield f"[TASK_START] {task_id}: {task}"
        
        try:
            result = await self.run_async(task, **kwargs)
            
            # Simulate streaming by yielding chunks
            result_str = str(result.result)
            chunk_size = 50
            
            for i in range(0, len(result_str), chunk_size):
                chunk = result_str[i:i+chunk_size]
                yield chunk
                await asyncio.sleep(0.1)  # Simulate processing time
            
            yield f"\n[TASK_COMPLETE] {task_id}: Success"
            
        except Exception as e:
            yield f"\n[TASK_ERROR] {task_id}: {str(e)}"
    
    def _handle_error(self, error: Exception, task: str, task_id: str) -> str:
        """Handle errors with recovery attempts."""
        self.logger.warning(f"Error in task {task_id}: {error}")
        
        # Check if it's an API authentication error
        if "403" in str(error) or "401" in str(error):
            return f"API Authentication Error: {str(error)}\n\nPlease check your API key and ensure it's valid and has the necessary permissions."
        elif "429" in str(error):
            return f"Rate Limit Error: {str(error)}\n\nPlease wait a moment and try again."
        
        try:
            # Try to get AI assistance for error recovery (only if API is working)
            recovery_prompt = f"""
            An error occurred while executing this task: {task}
            
            Error: {str(error)}
            
            Please provide suggestions for fixing this error or an alternative approach.
            """
            
            response = self.client.generate_completion(
                prompt=recovery_prompt,
                temperature=0.3
            )
            
            recovery_suggestion = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return f"Original error: {str(error)}\n\nRecovery suggestion: {recovery_suggestion}"
            
        except Exception as recovery_error:
            return f"Original error: {str(error)}\nRecovery attempt failed: {str(recovery_error)}"
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics for all tools."""
        return {tool_name: tool.get_stats() for tool_name, tool in self.tools.items()}
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of agent memory."""
        return {
            "agent_id": self.agent_id,
            "total_memories": len(self.memory.memories),
            "recent_memories": self.memory.get_recent_memories(5)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "agent_id": self.agent_id,
            "config": {
                "name": self.config.name,
                "model": self.config.model,
                "temperature": self.config.temperature
            },
            "usage_stats": self.usage_stats,
            "tool_stats": self.get_tool_stats(),
            "task_history_count": len(self.task_history),
            "memory_stats": {
                "total_memories": len(self.memory.memories),
                "max_memories": self.memory.max_memories
            },
            "success_rate": (
                self.usage_stats["successful_tasks"] / max(self.usage_stats["total_tasks"], 1)
            ) * 100,
            "average_task_time": (
                self.usage_stats["total_time"] / max(self.usage_stats["total_tasks"], 1)
            )
        }
    
    def _get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats.copy()
    
    def export_history(self, format: str = "json") -> str:
        """Export task history in specified format."""
        if format.lower() == "json":
            return json.dumps([
                {
                    "task_id": task.task_id,
                    "task": task.task,
                    "result": str(task.result),
                    "success": task.success,
                    "execution_time": task.execution_time,
                    "timestamp": task.timestamp.isoformat()
                }
                for task in self.task_history
            ], indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear_history(self) -> None:
        """Clear task history and reset statistics."""
        self.task_history.clear()
        self.usage_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "total_time": 0.0,
            "api_calls": 0,
            "errors": 0
        }
        self.memory.clear_memories()
        self.logger.info("History and statistics cleared")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information."""
        return {
            "agent_id": self.agent_id,
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_iterations": self.config.max_iterations,
                "timeout": self.config.timeout,
                "enable_memory": self.config.enable_memory,
                "enable_streaming": self.config.enable_streaming,
                "error_recovery": self.config.error_recovery
            },
            "tools": list(self.tools.keys()),
            "performance": self.get_performance_metrics()
        }


class EuriaiAgentBuilder:
    """Builder for creating specialized agents with pre-configured patterns."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = EuriaiClient(api_key=api_key)
    
    def create_coder_agent(self, 
                          name: str = "Coder Agent",
                          model: str = "gpt-4o",
                          temperature: float = 0.3) -> EuriaiSmolAgent:
        """Create a specialized coding agent."""
        config = AgentConfig(
            name=name,
            description="A specialized agent for coding tasks",
            model=model,
            temperature=temperature,
            max_iterations=15,
            enable_memory=True,
            error_recovery=True
        )
        
        agent = EuriaiSmolAgent(self.api_key, config)
        
        # Add coding-specific tools
        @tool
        def analyze_code(code: str) -> str:
            """Analyze code for potential improvements."""
            prompt = f"""
            Analyze this code for potential improvements, bugs, and best practices:
            
            {code}
            
            Provide specific suggestions for improvement.
            """
            
            response = agent.client.generate_completion(
                prompt=prompt,
                temperature=0.3
            )
            
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        @tool
        def debug_code(code: str, error: str) -> str:
            """Debug code with error information."""
            prompt = f"""
            Debug this code that's producing an error:
            
            Code: {code}
            Error: {error}
            
            Provide a solution and explanation.
            """
            
            response = agent.client.generate_completion(
                prompt=prompt,
                temperature=0.3
            )
            
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        agent.add_tool(analyze_code)
        agent.add_tool(debug_code)
        
        return agent
    
    def create_researcher_agent(self, 
                               name: str = "Researcher Agent",
                               model: str = "gpt-4o",
                               temperature: float = 0.5) -> EuriaiSmolAgent:
        """Create a specialized research agent."""
        config = AgentConfig(
            name=name,
            description="A specialized agent for research tasks",
            model=model,
            temperature=temperature,
            max_iterations=20,
            enable_memory=True,
            error_recovery=True
        )
        
        agent = EuriaiSmolAgent(self.api_key, config)
        
        # Add research-specific tools
        @tool
        def summarize_research(text: str) -> str:
            """Summarize research content."""
            prompt = f"""
            Summarize this research content, highlighting key findings and insights:
            
            {text}
            
            Provide a concise summary with main points.
            """
            
            response = agent.client.generate_completion(
                prompt=prompt,
                temperature=0.5
            )
            
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        @tool
        def analyze_data(data: str) -> str:
            """Analyze data and provide insights."""
            prompt = f"""
            Analyze this data and provide insights:
            
            {data}
            
            Identify patterns, trends, and key insights.
            """
            
            response = agent.client.generate_completion(
                prompt=prompt,
                temperature=0.5
            )
            
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        agent.add_tool(summarize_research)
        agent.add_tool(analyze_data)
        
        return agent
    
    def create_creative_agent(self, 
                             name: str = "Creative Agent",
                             model: str = "gpt-4o",
                             temperature: float = 0.9) -> EuriaiSmolAgent:
        """Create a specialized creative agent."""
        config = AgentConfig(
            name=name,
            description="A specialized agent for creative tasks",
            model=model,
            temperature=temperature,
            max_iterations=12,
            enable_memory=True,
            error_recovery=True
        )
        
        agent = EuriaiSmolAgent(self.api_key, config)
        
        # Add creative-specific tools
        @tool
        def generate_ideas(topic: str) -> str:
            """Generate creative ideas for a topic."""
            prompt = f"""
            Generate creative and innovative ideas for: {topic}
            
            Provide diverse, original, and practical ideas.
            """
            
            response = agent.client.generate_completion(
                prompt=prompt,
                temperature=0.9
            )
            
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        @tool
        def improve_content(content: str) -> str:
            """Improve and enhance content creatively."""
            prompt = f"""
            Improve this content to make it more engaging and creative:
            
            {content}
            
            Enhance style, flow, and impact while maintaining the core message.
            """
            
            response = agent.client.generate_completion(
                prompt=prompt,
                temperature=0.9
            )
            
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        agent.add_tool(generate_ideas)
        agent.add_tool(improve_content)
        
        return agent
    
    def create_multi_model_agent(self, 
                                name: str = "Multi-Model Agent",
                                models: List[str] = None) -> EuriaiSmolAgent:
        """Create an agent that can use multiple models."""
        if models is None:
            models = ["gpt-4o", "claude-3-sonnet-20240229", "gemini-pro"]
        
        config = AgentConfig(
            name=name,
            description="An agent that can use multiple AI models",
            model=models[0],  # Default model
            temperature=0.7,
            max_iterations=15,
            enable_memory=True,
            error_recovery=True
        )
        
        agent = EuriaiSmolAgent(self.api_key, config)
        
        # Add multi-model tools
        @tool
        def compare_models(prompt: str) -> str:
            """Compare responses from different models."""
            results = {}
            
            for model in models:
                try:
                    response = agent.client.generate_completion(
                        prompt=prompt,
                        temperature=0.7
                    )
                    results[model] = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                except Exception as e:
                    results[model] = f"Error: {str(e)}"
            
            comparison = "Model Comparison Results:\n\n"
            for model, result in results.items():
                comparison += f"=== {model} ===\n{result}\n\n"
            
            return comparison
        
        @tool
        def switch_model(new_model: str) -> str:
            """Switch the agent's default model."""
            if new_model in models:
                agent.config.model = new_model
                return f"Switched to model: {new_model}"
            else:
                return f"Model {new_model} not available. Available models: {', '.join(models)}"
        
        agent.add_tool(compare_models)
        agent.add_tool(switch_model)
        
        return agent


# Pre-defined agent patterns
def create_agent_pattern(api_key: str, 
                        agent_type: AgentType, 
                        name: Optional[str] = None,
                        **kwargs) -> EuriaiSmolAgent:
    """Create an agent using pre-defined patterns."""
    builder = EuriaiAgentBuilder(api_key)
    
    if agent_type == AgentType.CODER:
        return builder.create_coder_agent(name or "Coder Agent", **kwargs)
    elif agent_type == AgentType.RESEARCHER:
        return builder.create_researcher_agent(name or "Researcher Agent", **kwargs)
    elif agent_type == AgentType.CREATIVE:
        return builder.create_creative_agent(name or "Creative Agent", **kwargs)
    elif agent_type == AgentType.MULTI_TOOL:
        return builder.create_multi_model_agent(name or "Multi-Model Agent", **kwargs)
    else:
        # Default agent
        config = AgentConfig(
            name=name or "General Agent",
            description=f"A {agent_type.value} agent",
            **kwargs
        )
        return EuriaiSmolAgent(api_key, config)


# Utility functions
def ai_tool(description: str, model: str = "gpt-4o"):
    """Decorator to create AI-enhanced tools."""
    def decorator(func):
        func._needs_ai = True
        func._ai_model = model
        func._ai_description = description
        return func
    return decorator


@ai_tool("Analyze sentiment of text")
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text (AI-enhanced)."""
    # This will be enhanced with AI in EuriaiTool
    return f"Analyzing sentiment of: {text}"


@ai_tool("Translate text to another language")
def translate_text(text: str, target_language: str) -> str:
    """Translate text to another language (AI-enhanced)."""
    # This will be enhanced with AI in EuriaiTool
    return f"Translating '{text}' to {target_language}"


@ai_tool("Generate creative content")
def generate_content(topic: str, content_type: str = "article") -> str:
    """Generate creative content (AI-enhanced)."""
    # This will be enhanced with AI in EuriaiTool
    return f"Generating {content_type} about: {topic}" 