from typing import Dict, Any
from tools.base import BaseTool
from schema import ToolResult

class TerminateTool(BaseTool):
    """Tool for terminating the agent's execution."""
    
    name: str = "terminate"
    description: str = "Terminate the current task execution"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Reason for termination"
            },
            "summary": {
                "type": "string",
                "description": "Summary of what was accomplished"
            }
        },
        "required": ["reason"]
    }
    
    async def execute(self, reason: str, summary: str = "", **kwargs) -> ToolResult:
        """Execute termination."""
        return ToolResult.success_result({
            "action": "terminate",
            "reason": reason,
            "summary": summary,
            "message": f"Task terminated: {reason}"
        })