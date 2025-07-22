# llm.py
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from schema import Message, ToolCall
import openai
import google.generativeai as genai

class LLMConfig(BaseModel):
    provider: str = "openai"  # "openai" or "gemini"
    model: str = "gpt-4o"
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.0

class LLM:
    """Language Model interface supporting OpenAI and Gemini."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate client based on provider."""
        if self.config.provider == "openai":
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        elif self.config.provider == "gemini":
            genai.configure(api_key=self.config.api_key)
            self._client = genai.GenerativeModel(self.config.model)
    
    async def ask(self, messages: List[Message], **kwargs) -> str:
        """Send a conversation request to the LLM."""
        if self.config.provider == "openai":
            return await self._ask_openai(messages, **kwargs)
        elif self.config.provider == "gemini":
            return await self._ask_gemini(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def ask_tool(self, messages: List[Message], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Send a tool-enabled request to the LLM."""
        if self.config.provider == "openai":
            return await self._ask_tool_openai(messages, tools, **kwargs)
        elif self.config.provider == "gemini":
            return await self._ask_tool_gemini(messages, tools, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def _ask_openai(self, messages: List[Message], **kwargs) -> str:
        """OpenAI implementation."""
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=formatted_messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    async def _ask_tool_openai(self, messages: List[Message], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """OpenAI tool calling implementation."""
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=formatted_messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **kwargs
        )
        
        message = response.choices[0].message
        return {
            "content": message.content,
            "tool_calls": message.tool_calls
        }
    
    async def _ask_gemini(self, messages: List[Message], **kwargs) -> str:
        """Gemini implementation."""
        # Convert messages to Gemini format
        conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, self._client.generate_content, conversation_text
        )
        
        return response.text
    
    async def _ask_tool_gemini(self, messages: List[Message], tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Gemini tool calling implementation."""
        # Simplified implementation - in reality, you'd need to format tools for Gemini
        conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, self._client.generate_content, conversation_text
        )
        
        return {
            "content": response.text,
            "tool_calls": None  # Gemini tool calling would need proper implementation
        }