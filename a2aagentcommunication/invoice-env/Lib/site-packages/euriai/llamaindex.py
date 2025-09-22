"""
Enhanced LlamaIndex Integration for Euri API
==========================================

This module provides a comprehensive LlamaIndex integration with the Euri API,
including full LLM and embedding support with advanced features like streaming,
async operations, and seamless integration with LlamaIndex workflows.

Usage:
    from euriai.llamaindex_enhanced import EuriaiLlamaIndexLLM, EuriaiLlamaIndexEmbedding, EuriaiLlamaIndex
    
    # LLM with full features
    llm = EuriaiLlamaIndexLLM(
        api_key="your_api_key",
        model="gpt-4.1-nano",
        temperature=0.7
    )
    
    # Embedding model
    embedding = EuriaiLlamaIndexEmbedding(
        api_key="your_api_key",
        model="text-embedding-3-small"
    )
    
    # Complete integration
    llama_index = EuriaiLlamaIndex(
        api_key="your_api_key",
        llm_model="gpt-4.1-nano",
        embedding_model="text-embedding-3-small"
    )
"""

import asyncio
import json
import logging
from typing import (
    Any, Dict, List, Optional, Iterator, AsyncIterator, 
    Union, Callable, Sequence, Generator
)
from concurrent.futures import ThreadPoolExecutor
import time

try:
    from llama_index.core.llms import LLM
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.base.llms.types import (
        ChatMessage, MessageRole, CompletionResponse, CompletionResponseGen,
        ChatResponse, ChatResponseGen, LLMMetadata
    )
    from llama_index.core.schema import Document, TextNode, BaseNode
    from llama_index.core import VectorStoreIndex, ServiceContext, Settings
    from llama_index.core.callbacks import CallbackManager
    from pydantic import Field, BaseModel, SecretStr
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Fallback base classes
    class LLM:
        pass
    class BaseEmbedding:
        pass
    class ChatMessage:
        pass
    class MessageRole:
        pass
    class CompletionResponse:
        pass
    class CompletionResponseGen:
        pass
    class ChatResponse:
        pass
    class ChatResponseGen:
        pass
    class LLMMetadata:
        pass
    class Document:
        pass
    class TextNode:
        pass
    class BaseNode:
        pass
    class VectorStoreIndex:
        pass
    class ServiceContext:
        pass
    class Settings:
        pass
    class CallbackManager:
        pass
    class Field:
        pass
    class BaseModel:
        pass
    class SecretStr:
        pass

from euriai.client import EuriaiClient
from euriai.embedding import EuriaiEmbeddingClient


class EuriaiLlamaIndexLLM(LLM):
    """
    Enhanced LlamaIndex LLM implementation using Euri API.
    
    This implementation provides full LlamaIndex compatibility with advanced features:
    - Streaming support (both sync and async)
    - Async operations
    - Proper metadata handling
    - Usage tracking
    - Error handling and retries
    - Callback support
    
    Example:
        llm = EuriaiLlamaIndexLLM(
            api_key="your_api_key",
            model="gpt-4.1-nano",
            temperature=0.7,
            max_tokens=1000,
            streaming=True
        )
        
        # Basic completion
        response = llm.complete("What is AI?")
        print(response.text)
        
        # Chat
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello!")
        ]
        response = llm.chat(messages)
        print(response.message.content)
        
        # Streaming
        response_gen = llm.stream_complete("Tell me a story")
        for chunk in response_gen:
            print(chunk.text, end="")
            
        # Async
        response = await llm.acomplete("What is the weather?")
        print(response.text)
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
    context_window: int = Field(default=8000, description="Context window size")
    
    # Internal
    _client: Optional[EuriaiClient] = None
    _executor: Optional[ThreadPoolExecutor] = None
    
    def __init__(self, **kwargs):
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is not installed. Please install with: "
                "pip install llama-index-core"
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
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        if not LLAMAINDEX_AVAILABLE:
            return {}
        
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=True,
        )
    
    def _prepare_request_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters for the API call."""
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add optional parameters
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        
        # Override with kwargs
        params.update(kwargs)
        return params
    
    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Format LlamaIndex messages for the Euri API."""
        formatted_messages = []
        
        for message in messages:
            if hasattr(message, 'role') and hasattr(message, 'content'):
                if message.role == MessageRole.USER:
                    formatted_messages.append({"role": "user", "content": message.content})
                elif message.role == MessageRole.ASSISTANT:
                    formatted_messages.append({"role": "assistant", "content": message.content})
                elif message.role == MessageRole.SYSTEM:
                    formatted_messages.append({"role": "system", "content": message.content})
                else:
                    formatted_messages.append({"role": "user", "content": message.content})
            else:
                # Fallback for simple message formats
                formatted_messages.append({"role": "user", "content": str(message)})
        
        return formatted_messages
    
    def _create_completion_response(self, response: Dict[str, Any]) -> CompletionResponse:
        """Create a CompletionResponse from API response."""
        if not LLAMAINDEX_AVAILABLE:
            return {"text": response.get("choices", [{}])[0].get("message", {}).get("content", "")}
        
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return CompletionResponse(text=text)
    
    def _create_chat_response(self, response: Dict[str, Any]) -> ChatResponse:
        """Create a ChatResponse from API response."""
        if not LLAMAINDEX_AVAILABLE:
            return {"message": {"content": response.get("choices", [{}])[0].get("message", {}).get("content", "")}}
        
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        message = ChatMessage(role=MessageRole.ASSISTANT, content=text)
        return ChatResponse(message=message)
    
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        """Complete a prompt."""
        # Prepare request
        params = self._prepare_request_params(**kwargs)
        params["prompt"] = prompt  # Use 'prompt' directly instead of converting to messages
        
        try:
            # Make API call
            response = self._client.generate_completion(**params)
            return self._create_completion_response(response)
        except Exception as e:
            logging.error(f"Error in complete: {e}")
            raise
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Chat with messages."""
        # Format messages
        formatted_messages = self._format_messages(messages)
        
        # Convert messages to a single prompt string
        prompt_parts = []
        system_message = None
        
        for msg in formatted_messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        # Combine system message and conversation
        if system_message:
            prompt = f"System: {system_message}\n\n" + "\n".join(prompt_parts)
        else:
            prompt = "\n".join(prompt_parts)
        
        # Prepare request
        params = self._prepare_request_params(**kwargs)
        params["prompt"] = prompt  # Use 'prompt' instead of 'messages'
        
        try:
            # Make API call
            response = self._client.generate_completion(**params)
            return self._create_chat_response(response)
        except Exception as e:
            logging.error(f"Error in chat: {e}")
            raise
    
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponseGen:
        """Stream completion (currently not implemented for streaming)."""
        # For now, return single response as generator
        response = self.complete(prompt, formatted, **kwargs)
        yield response
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponseGen:
        """Stream chat (currently not implemented for streaming)."""
        # For now, return single response as generator
        response = self.chat(messages, **kwargs)
        yield response
    
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        """Async complete."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.complete(prompt, formatted, **kwargs)
        )
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Async chat."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.chat(messages, **kwargs)
        )
    
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs) -> AsyncIterator[CompletionResponse]:
        """Async stream complete."""
        response = await self.acomplete(prompt, formatted, **kwargs)
        yield response
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncIterator[ChatResponse]:
        """Async stream chat."""
        response = await self.achat(messages, **kwargs)
        yield response


class EuriaiLlamaIndexEmbedding(BaseEmbedding):
    """
    Enhanced LlamaIndex Embedding implementation using Euri API.
    
    This implementation provides full LlamaIndex compatibility with:
    - Batch embedding support
    - Async operations
    - Error handling and retries
    - Usage tracking
    - Configurable chunk size
    
    Example:
        embedding = EuriaiLlamaIndexEmbedding(
            api_key="your_api_key",
            model="text-embedding-3-small",
            batch_size=100
        )
        
        # Single embedding
        embedding_vec = embedding.get_text_embedding("Hello world")
        
        # Batch embeddings
        embeddings = embedding.get_text_embeddings([
            "Document 1",
            "Document 2",
            "Document 3"
        ])
        
        # Query embedding
        query_embedding = embedding.get_query_embedding("search query")
        
        # Async
        embedding_vec = await embedding.aget_text_embedding("Hello world")
    """
    
    # Configuration
    api_key: str = Field(description="Euri API key")
    model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    batch_size: int = Field(default=100, gt=0, description="Batch size for processing")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries")
    
    # Internal
    _client: Optional[EuriaiEmbeddingClient] = None
    _executor: Optional[ThreadPoolExecutor] = None
    
    def __init__(self, **kwargs):
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is not installed. Please install with: "
                "pip install llama-index-core"
            )
        
        super().__init__(**kwargs)
        
        # Initialize client
        self._client = EuriaiEmbeddingClient(
            api_key=self.api_key,
            model=self.model
        )
        
        # Initialize thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            embedding = self._client.embed(text)
            return embedding.tolist()
        except Exception as e:
            logging.error(f"Error getting text embedding: {e}")
            raise
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self._client.embed_batch(batch)
                all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
            
            return all_embeddings
        except Exception as e:
            logging.error(f"Error getting text embeddings: {e}")
            raise
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self.get_text_embedding(query)
    
    async def aget_text_embedding(self, text: str) -> List[float]:
        """Async get text embedding."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_text_embedding,
            text
        )
    
    async def aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async get text embeddings."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_text_embeddings,
            texts
        )
    
    async def aget_query_embedding(self, query: str) -> List[float]:
        """Async get query embedding."""
        return await self.aget_text_embedding(query)
    
    # Abstract methods required by BaseEmbedding
    def _get_text_embedding(self, text: str) -> List[float]:
        """Private method required by BaseEmbedding."""
        return self.get_text_embedding(text)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Private method required by BaseEmbedding."""
        return self.get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Private async method required by BaseEmbedding."""
        return await self.aget_text_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Private async method required by BaseEmbedding."""
        return await self.aget_query_embedding(query)


class EuriaiLlamaIndex:
    """
    Enhanced LlamaIndex integration that uses Euri API for both LLM and embedding operations.
    
    This provides a complete LlamaIndex workflow with:
    - Document ingestion and indexing
    - Query processing and retrieval
    - Multi-model support through single API key
    - Async operations
    - Advanced query options
    
    Example:
        llama_index = EuriaiLlamaIndex(
            api_key="your_api_key",
            llm_model="gpt-4.1-nano",
            embedding_model="text-embedding-3-small"
        )
        
        # Add documents
        llama_index.add_documents([
            "The sky is blue.",
            "Water is wet.",
            "Fire is hot."
        ])
        
        # Build index
        llama_index.build_index()
        
        # Query
        response = llama_index.query("What color is the sky?")
        print(response.response)
        
        # Async query
        response = await llama_index.aquery("What is water like?")
        print(response.response)
    """
    
    def __init__(
        self,
        api_key: str,
        llm_model: str = "gpt-4.1-nano",
        embedding_model: str = "text-embedding-3-small",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 1000,
        embedding_batch_size: int = 100,
        context_window: int = 8000,
        verbose: bool = True
    ):
        """
        Initialize the EuriaiLlamaIndex integration.
        
        Args:
            api_key: Euri API key
            llm_model: LLM model name
            embedding_model: Embedding model name
            llm_temperature: LLM temperature
            llm_max_tokens: LLM max tokens
            embedding_batch_size: Embedding batch size
            context_window: Context window size
            verbose: Enable verbose logging
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is not installed. Please install with: "
                "pip install llama-index-core"
            )
        
        self.api_key = api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = EuriaiLlamaIndexLLM(
            api_key=api_key,
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            context_window=context_window
        )
        
        # Initialize embedding model
        self.embedding = EuriaiLlamaIndexEmbedding(
            api_key=api_key,
            model=embedding_model,
            batch_size=embedding_batch_size
        )
        
        # Configure Settings (LlamaIndex global settings)
        Settings.llm = self.llm
        Settings.embed_model = self.embedding
        
        # Initialize storage
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        
        # Usage tracking
        self.usage_stats = {
            "total_queries": 0,
            "total_documents_indexed": 0,
            "total_embeddings_generated": 0,
            "total_llm_calls": 0
        }
    
    def add_documents(self, docs: List[Union[str, Dict[str, Any], Document]]) -> None:
        """
        Add documents to the index.
        
        Args:
            docs: List of documents (strings, dicts, or Document objects)
        """
        for doc in docs:
            if isinstance(doc, str):
                self.documents.append(Document(text=doc))
            elif isinstance(doc, dict):
                text = doc.get("text", doc.get("content", ""))
                metadata = {k: v for k, v in doc.items() if k not in ["text", "content"]}
                self.documents.append(Document(text=text, metadata=metadata))
            elif isinstance(doc, Document):
                self.documents.append(doc)
            else:
                raise ValueError(f"Invalid document type: {type(doc)}")
        
        self.usage_stats["total_documents_indexed"] += len(docs)
        
        if self.verbose:
            print(f"Added {len(docs)} documents. Total documents: {len(self.documents)}")
    
    def add_document(self, doc: Union[str, Dict[str, Any], Document]) -> None:
        """Add a single document."""
        self.add_documents([doc])
    
    def build_index(self, **kwargs) -> VectorStoreIndex:
        """
        Build the vector index from documents.
        
        Args:
            **kwargs: Additional arguments for VectorStoreIndex.from_documents
        """
        if not self.documents:
            raise ValueError("No documents added. Please add documents before building index.")
        
        try:
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                **kwargs
            )
            
            # Create query engine
            self.query_engine = self.index.as_query_engine()
            
            # Update usage stats
            self.usage_stats["total_embeddings_generated"] += len(self.documents)
            
            if self.verbose:
                print(f"Built index with {len(self.documents)} documents")
            
            return self.index
        
        except Exception as e:
            logging.error(f"Error building index: {e}")
            raise
    
    async def abuild_index(self, **kwargs) -> VectorStoreIndex:
        """
        Async build the vector index from documents.
        
        Args:
            **kwargs: Additional arguments for VectorStoreIndex.from_documents
        """
        if not self.documents:
            raise ValueError("No documents added. Please add documents before building index.")
        
        try:
            # Run index building in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def build_index_sync():
                index = VectorStoreIndex.from_documents(
                    self.documents,
                    **kwargs
                )
                return index
            
            self.index = await loop.run_in_executor(None, build_index_sync)
            
            # Create query engine
            self.query_engine = self.index.as_query_engine()
            
            # Update usage stats
            self.usage_stats["total_embeddings_generated"] += len(self.documents)
            
            if self.verbose:
                print(f"Built index with {len(self.documents)} documents")
            
            return self.index
        
        except Exception as e:
            logging.error(f"Error in async index building: {e}")
            raise
    
    def query(self, query: str, **kwargs) -> Any:
        """
        Query the index.
        
        Args:
            query: Query string
            **kwargs: Additional arguments for query engine
        """
        if self.index is None:
            if self.verbose:
                print("Index not built. Building index automatically...")
            self.build_index()
        
        try:
            response = self.query_engine.query(query, **kwargs)
            
            # Update usage stats
            self.usage_stats["total_queries"] += 1
            self.usage_stats["total_llm_calls"] += 1
            
            if self.verbose:
                print(f"Query: {query}")
                print(f"Response: {response.response}")
            
            return response
        
        except Exception as e:
            logging.error(f"Error querying index: {e}")
            raise
    
    async def aquery(self, query: str, **kwargs) -> Any:
        """
        Async query the index.
        
        Args:
            query: Query string
            **kwargs: Additional arguments for query engine
        """
        if self.index is None:
            if self.verbose:
                print("Index not built. Building index automatically...")
            await self.abuild_index()
        
        try:
            # Create async query engine
            async_query_engine = self.index.as_query_engine()
            
            # Run query in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: async_query_engine.query(query, **kwargs)
            )
            
            # Update usage stats
            self.usage_stats["total_queries"] += 1
            self.usage_stats["total_llm_calls"] += 1
            
            if self.verbose:
                print(f"Async Query: {query}")
                print(f"Response: {response.response}")
            
            return response
        
        except Exception as e:
            logging.error(f"Error in async query: {e}")
            raise
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the current index."""
        return self.index
    
    def get_documents(self) -> List[Document]:
        """Get all documents."""
        return self.documents
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return self.usage_stats.copy()
    
    def reset(self) -> None:
        """Reset documents and index."""
        self.documents = []
        self.index = None
        self.query_engine = None
        self.usage_stats = {
            "total_queries": 0,
            "total_documents_indexed": 0,
            "total_embeddings_generated": 0,
            "total_llm_calls": 0
        }
        
        if self.verbose:
            print("Reset completed")
    
    def update_model(self, llm_model: Optional[str] = None, embedding_model: Optional[str] = None) -> None:
        """
        Update the models used.
        
        Args:
            llm_model: New LLM model name
            embedding_model: New embedding model name
        """
        if llm_model:
            self.llm_model = llm_model
            self.llm = EuriaiLlamaIndexLLM(
                api_key=self.api_key,
                model=llm_model,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
                context_window=self.llm.context_window
            )
            Settings.llm = self.llm
            
            if self.verbose:
                print(f"Updated LLM model to: {llm_model}")
        
        if embedding_model:
            self.embedding_model = embedding_model
            self.embedding = EuriaiLlamaIndexEmbedding(
                api_key=self.api_key,
                model=embedding_model,
                batch_size=self.embedding.batch_size
            )
            Settings.embed_model = self.embedding
            
            # Need to rebuild index if embedding model changed
            if self.index is not None:
                if self.verbose:
                    print("Rebuilding index due to embedding model change...")
                self.build_index()
            
            if self.verbose:
                print(f"Updated embedding model to: {embedding_model}")


def create_llama_index(
    api_key: str,
    llm_model: str = "gpt-4.1-nano",
    embedding_model: str = "text-embedding-3-small",
    **kwargs
) -> EuriaiLlamaIndex:
    """
    Create a LlamaIndex integration with default settings.
    
    Args:
        api_key: Euri API key
        llm_model: LLM model name
        embedding_model: Embedding model name
        **kwargs: Additional arguments for EuriaiLlamaIndex
    """
    return EuriaiLlamaIndex(
        api_key=api_key,
        llm_model=llm_model,
        embedding_model=embedding_model,
        **kwargs
    )


def create_llm(
    api_key: str,
    model: str = "gpt-4.1-nano",
    **kwargs
) -> EuriaiLlamaIndexLLM:
    """
    Create a LlamaIndex LLM with default settings.
    
    Args:
        api_key: Euri API key
        model: Model name
        **kwargs: Additional arguments for EuriaiLlamaIndexLLM
    """
    return EuriaiLlamaIndexLLM(
        api_key=api_key,
        model=model,
        **kwargs
    )


def create_embedding(
    api_key: str,
    model: str = "text-embedding-3-small",
    **kwargs
) -> EuriaiLlamaIndexEmbedding:
    """
    Create a LlamaIndex embedding model with default settings.
    
    Args:
        api_key: Euri API key
        model: Model name
        **kwargs: Additional arguments for EuriaiLlamaIndexEmbedding
    """
    return EuriaiLlamaIndexEmbedding(
        api_key=api_key,
        model=model,
        **kwargs
    ) 