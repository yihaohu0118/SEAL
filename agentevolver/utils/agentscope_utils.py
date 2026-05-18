# -*- coding: utf-8 -*-
"""Utilities for agentscope integration."""
import asyncio
import importlib
from typing import Dict, List, Any, Callable
from abc import ABC, abstractmethod

from loguru import logger

from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory
# Try to import agentscope model classes

from agentscope.model import ChatModelBase
from agentscope.model import ChatResponse
from agentscope.message import TextBlock, Msg


def dynamic_import(module_class_str: str):
    """
    Dynamically import a class from a module.
    
    Args:
        module_class_str (str): String in format "module.path->ClassName"
        
    Returns:
        The imported class.
    """
    module_, class_ = module_class_str.split("->")
    protocol_cls = getattr(importlib.import_module(module_), class_)
    return protocol_cls


class AgentscopeModelWrapper:
    """
    A wrapper class that adapts llm_chat function to agentscope's ChatModelBase interface.
    
    This wrapper converts between agentscope's message format (role/content with blocks) and
    the internal message format (role/content), and handles async/sync conversion.
    All input and output are based on block-based as message format.
    """
    
    def __init__(
        self,
        llm_chat_fn: Callable,
        model_name: str,
        stream: bool = False,
    ):
        """
        Initialize the AgentscopeModelWrapper.
        
        Args:
            llm_chat_fn (Callable): The llm_chat function from ParallelEnvManager.
            model_name (str): The name of the model.
            stream (bool): Whether to use streaming (currently not supported, defaults to False).
        """
   
        self.model_name = model_name
        self.stream = stream
        self.llm_chat_fn = llm_chat_fn
        
        # Store imported classes for use in methods
        self.ChatResponse = ChatResponse
        self.TextBlock = TextBlock
        self.Msg = Msg
    
    def _extract_text_from_content(self, content: Any) -> str:
        """
        Extract text content from agentscope message content (string or block list).
        
        This method handles block-based content format by extracting text from all text blocks.
        
        Args:
            content: Content can be:
                - str: Direct text content
                - list: List of content blocks (TextBlock, ImageBlock, etc.)
                    Each block can be:
                    - dict with "type" and "text" keys
                    - Object with "type" and "text" attributes (TextBlock)
                    - str (direct string in list)
        
        Returns:
            str: Extracted text content from all text blocks.
        """
        # Handle string content
        if isinstance(content, str):
            return content
        
        # Handle list of blocks
        if isinstance(content, list):
            text_parts = []
            for block in content:
                # Handle dict-based blocks
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    # For other block types (image, audio, video, etc.), 
                    # we skip them as they don't have text content
                # Handle object-based blocks (TextBlock, etc.)
                elif hasattr(block, "get"):
                    # Dict-like object
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    # Object with text attribute (TextBlock)
                    text_parts.append(block.text)
                elif isinstance(block, str):
                    # Direct string in list
                    text_parts.append(block)
            
            return "".join(text_parts)
        
        # Fallback: convert to string
        return str(content) if content else ""
    
    def _convert_messages_to_internal_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert agentscope message format (role/content with blocks) to internal format (role/content).
        
        This method properly handles block-based message format by extracting text from blocks.
        
        Args:
            messages (List[Dict[str, Any]]): Messages in agentscope format.
                Each message can be:
                - Dict with "role" and "content" keys, where content can be str or list of blocks
                - Msg object (has get_text_content method)
            
        Returns:
            List[Dict[str, str]]: Messages in internal format with "role" and "content" keys.
        """
        converted = []
        for msg in messages:
            # Handle Msg object (has get_text_content method)
            if hasattr(msg, "get_text_content"):
                # It's a Msg object, use its methods to extract role and content
                role = getattr(msg, "role", "user")
                content = msg.get_text_content() or ""
            else:
                # Handle dict format
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Extract text from content (handles both string and block formats)
                content = self._extract_text_from_content(content)
            
            converted.append({
                "role": role,
                "content": content,
            })
        return converted
    
    def _convert_response_to_agentscope_format(self, response: Dict[str, Any]) -> Any:
        """
        Convert internal response format to agentscope ChatResponse with block-based format.
        
        Args:
            response (Dict[str, Any]): Response in internal format (role/content/tokens).
                May contain 'tokens' field with token objects (each has token_id attribute).
            
        Returns:
            ChatResponse: Response in agentscope format with TextBlock content.
                metadata contains 'tokens' if available.
        """
        content = response.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        # Extract tokens if available (for training consistency)
        tokens = response.get("tokens", None)
        metadata = None
        if tokens is not None:
            # Store tokens in metadata for later use in AgentscopeCMT
            # Convert token objects to list of token_ids for serialization
            token_ids = [t.token_id if hasattr(t, 'token_id') else t for t in tokens]
            metadata = {"tokens": token_ids, "original_tokens": tokens}
        
        # Create TextBlock from content (block-based format)
        text_block = self.TextBlock(type="text", text=content)
        
        # Create ChatResponse with block-based content
        chat_response = self.ChatResponse(
            content=[text_block],  # Block-based format
            usage=None,  # Usage information not available from llm_chat
            metadata=metadata,
        )
        return chat_response
    
    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Call the wrapped llm_chat function asynchronously.
        
        Args:
            messages (List[Dict[str, Any]]): Messages in agentscope format.
            tools (List[Dict] | None): Tools (not supported, ignored).
            tool_choice (str | None): Tool choice (not supported, ignored).
            **kwargs: Additional keyword arguments (e.g., custom_sampling_params, request_id).
            
        Returns:
            ChatResponse: The response in agentscope format.
        """
        # Convert messages to internal format
        internal_messages = self._convert_messages_to_internal_format(messages)
        
        # Extract custom_sampling_params and request_id from kwargs
        custom_sampling_params = kwargs.get("custom_sampling_params")
        request_id = kwargs.get("request_id")
        
        # Call llm_chat in a thread pool to make it async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llm_chat_fn(
                messages=internal_messages,
                custom_sampling_params=custom_sampling_params,
                request_id=request_id,
            )
        )
        
        # Convert response to agentscope format
        return self._convert_response_to_agentscope_format(response)


class BaseAgentscopeWorkflow(ABC):
    """
    Base class for agentscope workflows.
    
    This class provides a standard interface for workflows that use agentscope agents.
    Subclasses should implement the execute() method to run the workflow and return a Trajectory.
    """
    
    def __init__(
        self,
        task: Task,
        llm_chat_fn: Callable,
        model_name: str,
        config: Any,
        tokenizer: Any,
        data_id: str,
        rollout_id: str,
        **kwargs
    ):
        """
        Initialize the workflow.
        
        Args:
            task (Task): The task to be executed.
            llm_chat_fn (Callable): The LLM chat function to use for agent creation.
            model_name (str): The name of the model.
            config: Configuration object (required for CMT functionality).
            tokenizer: Tokenizer instance (required for CMT functionality).
            data_id (str): The ID of the data.
            rollout_id (str): The ID of the rollout.
            **kwargs: Additional keyword arguments.
        """
        self.task = task
        self.llm_chat_fn = llm_chat_fn
        self.model_name = model_name
        self.config = config
        self.tokenizer = tokenizer
        self.data_id = data_id
        self.rollout_id = rollout_id
        
        # Create agentscope model wrapper from llm_chat_fn
        self.model = AgentscopeModelWrapper(
            llm_chat_fn=llm_chat_fn,
            model_name=model_name,
            stream=False,
        )
        
        # Agents will be created by subclasses
        self.agents = []
    
    @abstractmethod
    def execute(self) -> Trajectory:
        """
        Execute the workflow and return a Trajectory.
        
        Returns:
            Trajectory: The trajectory containing model call history and workflow results.
        """
        raise NotImplementedError

