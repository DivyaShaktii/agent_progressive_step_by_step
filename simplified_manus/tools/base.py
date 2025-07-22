from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from schema import ToolResult

class BaseTool(ABC, BaseModel):
    """Base class for all tools."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def to_param(self) -> Dict[str, Any]:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

class ToolCollection:
    """A collection of tools for easy management."""
    
    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
    
    def __iter__(self):
        return iter(self.tools)
    
    def to_params(self) -> List[Dict[str, Any]]:
        """Convert all tools to function call format."""
        return [tool.to_param() for tool in self.tools]
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.tool_map.get(name)
        if not tool:
            return ToolResult.error_result(f"Tool {name} not found")
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            return ToolResult.error_result(f"Tool execution failed: {str(e)}")