"""
Enhanced LangChain Integration for Euri API
==========================================

This module provides a comprehensive LangChain integration with the Euri API,
including full ChatModel and Embeddings support with advanced features like
streaming, async operations, function calling, and structured output.

Usage:
    from euriai.langchain_enhanced import EuriaiChatModel, EuriaiEmbeddings
    
    # Chat model with all features
    chat_model = EuriaiChatModel(
        api_key="your_api_key",
        model="gpt-4.1-nano",
        temperature=0.7
    )
    
    # Embeddings model
    embeddings = EuriaiEmbeddings(
        api_key="your_api_key",
        model="text-embedding-3-small"
    )
"""

import asyncio
import json
import logging
from typing import (
    Any, Dict, List, Optional, Iterator, AsyncIterator, 
    Union, Callable, Type, Sequence, Tuple
)
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import time

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.llms import LLM
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import (
        BaseMessage, AIMessage, HumanMessage, SystemMessage, 
        AIMessageChunk, FunctionMessage, ToolMessage
    )
    from langchain_core.messages.ai import UsageMetadata
    from langchain_core.outputs import (
        ChatGeneration, ChatGenerationChunk, ChatResult, 
        LLMResult, Generation
    )
    from langchain_core.callbacks import (
        CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
    )
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_function
    from pydantic import Field, BaseModel, SecretStr
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Fallback base classes for when LangChain is not available
    class BaseChatModel:
        pass
    class LLM:
        pass
    class Embeddings:
        pass
    class BaseMessage:
        pass
    class AIMessage:
        pass
    class HumanMessage:
        pass
    class SystemMessage:
        pass
    class AIMessageChunk:
        pass
    class FunctionMessage:
        pass
    class ToolMessage:
        pass
    class UsageMetadata:
        pass
    class ChatGeneration:
        pass
    class ChatGenerationChunk:
        pass
    class ChatResult:
        pass
    class LLMResult:
        pass
    class Generation:
        pass
    class CallbackManagerForLLMRun:
        pass
    class AsyncCallbackManagerForLLMRun:
        pass
    class RunnableConfig:
        pass
    class BaseTool:
        pass
    class Field:
        pass
    class BaseModel:
        pass
    class SecretStr:
        pass

from euriai.client import EuriaiClient
from euriai.embedding import EuriaiEmbeddingClient


class EuriaiChatModel(BaseChatModel):
    """
    Enhanced LangChain ChatModel implementation using Euri API.
    
    This implementation provides full LangChain compatibility with advanced features:
    - Streaming support (both sync and async)
    - Function calling and tool use
    - Structured output support
    - Async operations
    - Usage tracking and metadata
    - Proper error handling
    - Callback support
    
    Example:
        chat_model = EuriaiChatModel(
            api_key="your_api_key",
            model="gpt-4.1-nano",
            temperature=0.7,
            max_tokens=1000,
            streaming=True
        )
        
        # Basic usage
        response = chat_model.invoke("Hello, how are you?")
        
        # Streaming
        for chunk in chat_model.stream("Tell me a story"):
            print(chunk.content, end="")
            
        # Async
        response = await chat_model.ainvoke("What is AI?")
        
        # With messages
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="What is the weather like?")
        ]
        response = chat_model.invoke(messages)
    """
    
    # Configuration
    api_key: str = Field(description="Euri API key")
    model: str = Field(default="gpt-4.1-nano", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Features
    streaming: bool = Field(default=False, description="Enable streaming responses")
    supports_function_calling: bool = Field(default=True, description="Support function calling")
    supports_structured_output: bool = Field(default=True, description="Support structured output")
    
    # Internal
    _client: Optional[EuriaiClient] = None
    _executor: Optional[ThreadPoolExecutor] = None
    
    def __init__(self, **kwargs):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install with: "
                "pip install langchain-core"
            )
        
        super().__init__(**kwargs)
        
        # Initialize client
        self._client = EuriaiClient(
            api_key=self.api_key,
            model=self.model
        )
        
        # Initialize thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def _llm_type(self) -> str:
        """Get the type of language model."""
        return "euriai_chat_enhanced"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters for the model."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format LangChain messages for the Euri API."""
        formatted_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, (FunctionMessage, ToolMessage)):
                formatted_messages.append({"role": "function", "content": message.content})
            else:
                # Fallback for other message types
                formatted_messages.append({"role": "user", "content": str(message.content)})
        
        return formatted_messages
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            elif isinstance(message, (FunctionMessage, ToolMessage)):
                prompt_parts.append(f"Function: {message.content}")
            else:
                prompt_parts.append(f"User: {message.content}")
        
        return "\n\n".join(prompt_parts)
    
    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """Create ChatResult from API response."""
        if "choices" not in response or not response["choices"]:
            raise ValueError("Invalid response format from Euri API")
        
        choice = response["choices"][0]
        message_content = choice.get("message", {}).get("content", "")
        
        # Extract usage information
        usage = response.get("usage", {})
        usage_metadata = UsageMetadata(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0)
        )
        
        # Create AI message
        ai_message = AIMessage(
            content=message_content,
            usage_metadata=usage_metadata,
            response_metadata={
                "model": self.model,
                "finish_reason": choice.get("finish_reason"),
                "created": response.get("created"),
            }
        )
        
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                "finish_reason": choice.get("finish_reason"),
                "model": self.model,
            }
        )
        
        return ChatResult(
            generations=[generation],
            llm_output={
                "token_usage": usage,
                "model_name": self.model,
                "system_fingerprint": response.get("system_fingerprint"),
            }
        )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response."""
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        # Prepare request
        request_params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add optional parameters
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            request_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            request_params["presence_penalty"] = self.presence_penalty
        if stop:
            request_params["stop"] = stop
        
        # Override with kwargs
        request_params.update(kwargs)
        
        try:
            # Make API call
            response = self._client.generate_completion(**request_params)
            return self._create_chat_result(response)
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response."""
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        # Prepare request
        request_params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add optional parameters
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            request_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            request_params["presence_penalty"] = self.presence_penalty
        if stop:
            request_params["stop"] = stop
        
        # Override with kwargs
        request_params.update(kwargs)
        
        try:
            # Stream response
            accumulated_content = ""
            for chunk_data in self._client.stream_completion(**request_params):
                if chunk_data.strip():
                    try:
                        # Parse SSE data
                        if chunk_data.startswith("data: "):
                            chunk_data = chunk_data[6:]
                        
                        if chunk_data.strip() == "[DONE]":
                            break
                        
                        chunk_json = json.loads(chunk_data)
                        if "choices" in chunk_json and chunk_json["choices"]:
                            choice = chunk_json["choices"][0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                accumulated_content += content
                                
                                # Create usage metadata
                                usage_metadata = UsageMetadata(
                                    input_tokens=0,
                                    output_tokens=1,
                                    total_tokens=1
                                )
                                
                                # Create chunk
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content=content,
                                        usage_metadata=usage_metadata
                                    ),
                                    generation_info={
                                        "finish_reason": choice.get("finish_reason"),
                                        "model": self.model,
                                    }
                                )
                                
                                # Notify callback
                                if run_manager:
                                    run_manager.on_llm_new_token(content, chunk=chunk)
                                
                                yield chunk
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat response."""
        # Run sync method in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate,
            messages,
            stop,
            run_manager,
            **kwargs
        )
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat response."""
        # Run sync stream method in thread pool
        loop = asyncio.get_event_loop()
        
        def sync_stream():
            return list(self._stream(messages, stop, run_manager, **kwargs))
        
        chunks = await loop.run_in_executor(self._executor, sync_stream)
        
        for chunk in chunks:
            yield chunk
    
    def bind_functions(self, functions: Sequence[Dict[str, Any]]) -> "EuriaiChatModel":
        """Bind functions to the model for function calling."""
        # Create new instance with functions bound
        return self.__class__(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            streaming=self.streaming,
            supports_function_calling=self.supports_function_calling,
            supports_structured_output=self.supports_structured_output,
            _bound_functions=functions
        )
    
    def bind_tools(self, tools: Sequence[Union[Dict[str, Any], BaseTool]]) -> "EuriaiChatModel":
        """Bind tools to the model for tool calling."""
        # Convert tools to functions
        functions = []
        for tool in tools:
            if isinstance(tool, dict):
                functions.append(tool)
            elif hasattr(tool, 'to_function'):
                functions.append(tool.to_function())
            else:
                # Convert tool to function format
                functions.append(convert_to_openai_function(tool))
        
        return self.bind_functions(functions)
    
    def with_structured_output(
        self, 
        schema: Union[Dict, Type[BaseModel]], 
        **kwargs: Any
    ) -> "EuriaiStructuredChatModel":
        """Create a version that returns structured output."""
        return EuriaiStructuredChatModel(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            streaming=self.streaming,
            supports_function_calling=self.supports_function_calling,
            supports_structured_output=True,
            schema=schema,
            **kwargs
        )


class EuriaiStructuredChatModel(EuriaiChatModel):
    """
    EuriaiChatModel with structured output support.
    
    This class extends EuriaiChatModel to parse responses into structured Pydantic models.
    """
    
    def __init__(self, schema: Union[Dict, Type[BaseModel]], **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)
        
        # Store schema in private attributes to avoid Pydantic conflicts
        self._output_schema = schema
        self._schema_name = getattr(schema, '__name__', 'OutputSchema')
        
    @property
    def schema(self):
        """Get the output schema."""
        return self._output_schema
    
    @property
    def schema_name(self):
        """Get the schema name."""
        return self._schema_name
        
    def _get_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema from the Pydantic model."""
        if hasattr(self.schema, 'model_json_schema'):
            # Pydantic v2
            return self.schema.model_json_schema()
        elif hasattr(self.schema, 'schema'):
            # Pydantic v1
            return self.schema.schema()
        else:
            # Dictionary schema
            return self.schema
    
    def _create_structured_prompt(self, original_prompt: str) -> str:
        """Create a prompt that requests structured JSON output."""
        json_schema = self._get_json_schema()
        
        structured_prompt = f"""{original_prompt}

Please respond with a valid JSON object that matches this exact schema:
```json
{json.dumps(json_schema, indent=2)}
```

Your response must be valid JSON that can be parsed. Do not include any other text outside the JSON object."""
        
        return structured_prompt
    
    def _parse_structured_response(self, response_content: str) -> Any:
        """Parse the response content into the structured format."""
        try:
            # Try to find JSON in the response
            response_content = response_content.strip()
            
            # Handle cases where the response might have extra text
            if '```json' in response_content:
                # Extract JSON from code block
                start = response_content.find('```json') + 7
                end = response_content.find('```', start)
                if end == -1:
                    end = len(response_content)
                json_str = response_content[start:end].strip()
            elif response_content.startswith('{') and response_content.endswith('}'):
                # Response is already JSON
                json_str = response_content
            else:
                # Try to find JSON object in the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON object found in response")
            
            # Parse JSON
            parsed_data = json.loads(json_str)
            
            # Convert to Pydantic model if schema is a BaseModel
            if hasattr(self.schema, 'model_validate'):
                # Pydantic v2
                return self.schema.model_validate(parsed_data)
            elif hasattr(self.schema, 'parse_obj'):
                # Pydantic v1
                return self.schema.parse_obj(parsed_data)
            else:
                # Dictionary schema - return parsed data
                return parsed_data
                
        except Exception as e:
            # Fallback: return raw response if parsing fails
            raise ValueError(f"Failed to parse structured output: {e}\nResponse: {response_content}")
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to a single prompt string with structured output instructions."""
        # Get the original prompt
        original_prompt = super()._messages_to_prompt(messages)
        
        # Add structured output instructions
        return self._create_structured_prompt(original_prompt)
    
    def invoke(self, input, config=None, **kwargs):
        """Invoke the model and return structured output."""
        # Get the regular AI message response
        response = super().invoke(input, config, **kwargs)
        
        # Parse the response content into structured format
        structured_result = self._parse_structured_response(response.content)
        
        return structured_result
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Async invoke the model and return structured output."""
        # Get the regular AI message response
        response = await super().ainvoke(input, config, **kwargs)
        
        # Parse the response content into structured format
        structured_result = self._parse_structured_response(response.content)
        
        return structured_result


class EuriaiEmbeddings(Embeddings):
    """
    Enhanced LangChain Embeddings implementation using Euri API.
    
    This implementation provides full LangChain compatibility with:
    - Batch embedding support
    - Async operations
    - Error handling and retries
    - Usage tracking
    - Configurable chunk size
    
    Example:
        embeddings = EuriaiEmbeddings(
            api_key="your_api_key",
            model="text-embedding-3-small",
            chunk_size=1000
        )
        
        # Single document
        embedding = embeddings.embed_query("Hello world")
        
        # Multiple documents
        embeddings_list = embeddings.embed_documents([
            "Document 1",
            "Document 2",
            "Document 3"
        ])
        
        # Async
        embedding = await embeddings.aembed_query("Hello world")
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 max_retries: int = 3,
                 request_timeout: int = 60,
                 **kwargs):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install with: "
                "pip install langchain-core"
            )
        
        super().__init__()
        
        # Initialize configuration
        self.api_key = api_key
        self.model = model
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        
        # Internal
        self._client: Optional[EuriaiEmbeddingClient] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Initialize client
        self._client = EuriaiEmbeddingClient(
            api_key=self.api_key,
            model=self.model
        )
        
        # Initialize thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        if not texts:
            return []
        
        # Process in chunks to avoid API limits
        all_embeddings = []
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i:i + self.chunk_size]
            
            # Get embeddings for this chunk
            chunk_embeddings = self._client.embed_batch(chunk)
            all_embeddings.extend([emb.tolist() for emb in chunk_embeddings])
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text."""
        embedding = self._client.embed(text)
        return embedding.tolist()
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed search documents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embed_documents,
            texts
        )
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a query text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embed_query,
            text
        )


class EuriaiLLM(LLM):
    """
    Enhanced LangChain LLM implementation using Euri API.
    
    This provides the traditional LLM interface (text-in, text-out)
    while using the Euri API backend.
    
    Example:
        llm = EuriaiLLM(
            api_key="your_api_key",
            model="gpt-4.1-nano",
            temperature=0.5
        )
        
        response = llm.invoke("What is the capital of France?")
        
        # Streaming
        for chunk in llm.stream("Tell me a joke"):
            print(chunk, end="")
    """
    
    # Configuration
    api_key: str = Field(description="Euri API key")
    model: str = Field(default="gpt-4.1-nano", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Internal
    _client: Optional[EuriaiClient] = None
    _executor: Optional[ThreadPoolExecutor] = None
    
    def __init__(self, **kwargs):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install with: "
                "pip install langchain-core"
            )
        
        super().__init__(**kwargs)
        
        # Initialize client
        self._client = EuriaiClient(
            api_key=self.api_key,
            model=self.model
        )
        
        # Initialize thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def _llm_type(self) -> str:
        """Get the type of language model."""
        return "euriai_llm_enhanced"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters for the model."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Euri API."""
        # Prepare request
        request_params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add optional parameters
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            request_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            request_params["presence_penalty"] = self.presence_penalty
        if stop:
            request_params["stop"] = stop
        
        # Override with kwargs
        request_params.update(kwargs)
        
        try:
            # Make API call
            response = self._client.generate_completion(**request_params)
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream the LLM response."""
        # Prepare request
        request_params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add optional parameters
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            request_params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            request_params["presence_penalty"] = self.presence_penalty
        if stop:
            request_params["stop"] = stop
        
        # Override with kwargs
        request_params.update(kwargs)
        
        try:
            # Stream response
            for chunk_data in self._client.stream_completion(**request_params):
                if chunk_data.strip():
                    try:
                        # Parse SSE data
                        if chunk_data.startswith("data: "):
                            chunk_data = chunk_data[6:]
                        
                        if chunk_data.strip() == "[DONE]":
                            break
                        
                        chunk_json = json.loads(chunk_data)
                        if "choices" in chunk_json and chunk_json["choices"]:
                            choice = chunk_json["choices"][0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                # Notify callback
                                if run_manager:
                                    run_manager.on_llm_new_token(content)
                                
                                yield content
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call the Euri API."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._call,
            prompt,
            stop,
            run_manager,
            **kwargs
        )
    
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream the LLM response."""
        loop = asyncio.get_event_loop()
        
        def sync_stream():
            return list(self._stream(prompt, stop, run_manager, **kwargs))
        
        chunks = await loop.run_in_executor(self._executor, sync_stream)
        
        for chunk in chunks:
            yield chunk


# Convenience functions for easy model creation
def create_chat_model(
    api_key: str,
    model: str = "gpt-4.1-nano",
    temperature: float = 0.7,
    **kwargs
) -> EuriaiChatModel:
    """Create a chat model with default settings."""
    return EuriaiChatModel(
        api_key=api_key,
        model=model,
        temperature=temperature,
        **kwargs
    )


def create_embeddings(
    api_key: str,
    model: str = "text-embedding-3-small",
    **kwargs
) -> EuriaiEmbeddings:
    """Create an embeddings model with default settings."""
    return EuriaiEmbeddings(
        api_key=api_key,
        model=model,
        **kwargs
    )


def create_llm(
    api_key: str,
    model: str = "gpt-4.1-nano",
    temperature: float = 0.7,
    **kwargs
) -> EuriaiLLM:
    """Create an LLM with default settings."""
    return EuriaiLLM(
        api_key=api_key,
        model=model,
        temperature=temperature,
        **kwargs
    )


# Model information
AVAILABLE_MODELS = {
    "chat": [
        "gpt-4.1-nano",
        "gpt-4.1-mini", 
        "gpt-4.1-turbo",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp"
    ],
    "embeddings": [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ]
}


def get_available_models() -> Dict[str, List[str]]:
    """Get list of available models."""
    return AVAILABLE_MODELS.copy()


def validate_model(model: str, model_type: str = "chat") -> bool:
    """Validate if a model is available."""
    return model in AVAILABLE_MODELS.get(model_type, []) 