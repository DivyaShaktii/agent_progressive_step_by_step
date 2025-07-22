from typing import List, Dict, Any, Optional
from agents.react import ReActAgent
from tools.base import ToolCollection, BaseTool
from tools.terminate import TerminateTool
from schema import Message, ToolCall, AgentState
from pydantic import Field

class ToolCallAgent(ReActAgent):
    """Agent that can call tools to perform actions."""
    
    available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(TerminateTool()))
    tool_calls: List[ToolCall] = Field(default_factory=list)
    
    async def think(self) -> bool:
        """Think and decide which tools to use."""
        messages = self.memory.get_messages()
        
        # Add system prompt if available
        if self.system_prompt:
            messages = [Message.system_message(self.system_prompt)] + messages
        
        # Get LLM response with tool capabilities
        response = await self.llm.ask_tool(
            messages=messages,
            tools=self.available_tools.to_params()
        )
        
        # Update memory with assistant response
        #self.update_memory("assistant", response.get("content", ""))
        self.update_memory("assistant", response.get("content") or "")
        
        # Extract tool calls
        tool_calls = response.get("tool_calls", [])
        self.tool_calls = [ToolCall(**call) for call in tool_calls] if tool_calls else []
        
        return len(self.tool_calls) > 0
    
    async def act(self) -> str:
        """Execute the selected tools."""
        if not self.tool_calls:
            return "No tools to execute"
        
        results = []
        for tool_call in self.tool_calls:
            try:
                # Execute the tool
                function_name = tool_call.function["name"]
                function_args = tool_call.function.get("arguments", {})
                
                # Parse arguments if they're a string
                if isinstance(function_args, str):
                    import json
                    function_args = json.loads(function_args)
                
                result = await self.available_tools.execute(function_name, **function_args)
                
                # Update memory with tool result
                self.update_memory("tool", str(result.output), tool_call_id=tool_call.id)
                
                results.append(f"Tool {function_name}: {result.output}")
                
                # Check if this is a termination
                if function_name == "terminate":
                    self.state = AgentState.COMPLETED
                
            except Exception as e:
                error_msg = f"Tool {function_name} failed: {str(e)}"
                results.append(error_msg)
                self.update_memory("tool", error_msg, tool_call_id=tool_call.id)
        
        return "\n".join(results)