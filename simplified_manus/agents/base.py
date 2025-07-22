from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from schema import AgentState, Message
from memory import Memory
from llm import LLM

class BaseAgent(ABC, BaseModel):
    """Abstract base class for all agents."""
    
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # Core components
    llm: LLM
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    
    # Execution control
    max_steps: int = 10
    current_step: int = 0
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    async def step(self) -> str:
        """Execute one step of the agent's logic."""
        pass
    
    async def run(self, initial_message: str) -> str:
        """Run the agent with an initial message."""
        self.state = AgentState.THINKING
        self.memory.add_message(Message.user_message(initial_message))
        
        result = ""
        while self.current_step < self.max_steps and self.state != AgentState.COMPLETED:
            try:
                step_result = await self.step()
                result = step_result
                self.current_step += 1
                
                # Check if we should stop
                if self.state == AgentState.COMPLETED:
                    break
                    
            except Exception as e:
                self.state = AgentState.ERROR
                result = f"Error: {str(e)}"
                break
        
        return result
    
    def update_memory(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the agent's memory."""
        if role == "user":
            message = Message.user_message(content)
        elif role == "assistant":
            message = Message.assistant_message(content, **kwargs)
        elif role == "system":
            message = Message.system_message(content)
        elif role == "tool":
            message = Message.tool_message(content, **kwargs)
        else:
            raise ValueError(f"Unknown role: {role}")
        
        self.memory.add_message(message)

