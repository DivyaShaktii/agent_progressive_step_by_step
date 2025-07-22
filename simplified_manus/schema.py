from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETED = "completed"
    ERROR = "error"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    @classmethod
    def user_message(cls, content: str) -> "Message":
        return cls(role="user", content=content)
    
    @classmethod
    def assistant_message(cls, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)
    
    @classmethod
    def system_message(cls, content: str) -> "Message":
        return cls(role="system", content=content)
    
    @classmethod
    def tool_message(cls, content: str, tool_call_id: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "function"
    function: Dict[str, Any]

class ToolResult(BaseModel):
    output: Any = None
    error: Optional[str] = None
    success: bool = True
    
    @classmethod
    def success_result(cls, output: Any) -> "ToolResult":
        return cls(output=output, success=True)
    
    @classmethod
    def error_result(cls, error: str) -> "ToolResult":
        return cls(error=error, success=False)

class Plan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    steps: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)