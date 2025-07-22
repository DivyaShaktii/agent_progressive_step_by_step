from typing import List, Optional
from pydantic import BaseModel, Field
from schema import Message

class Memory(BaseModel):
    """Stores and manages the agent's conversation history."""
    
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)
    
    def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        self.messages.append(message)
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Message]:
        """Get all messages in memory."""
        return self.messages
    
    def get_recent_messages(self, count: int) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if count > 0 else []
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages.clear()
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return "No conversation history"
        
        summary = []
        for msg in self.messages[-10:]:  # Last 10 messages
            summary.append(f"{msg.role}: {msg.content[:100]}...")
        return "\n".join(summary)