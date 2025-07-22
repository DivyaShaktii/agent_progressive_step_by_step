# tools/planning.py
from typing import Dict, Any, List, Optional
from tools.base import BaseTool
from schema import ToolResult, Plan
import json
from pydantic import Field
from pydantic import PrivateAttr

class PlanningTool(BaseTool):
    """Tool for creating and managing task plans."""
    
    name: str = "planning"
    description: str = "Create and manage plans for complex tasks"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["create", "update", "get", "list", "mark_step", "delete"],
                "description": "The command to execute"
            },
            "plan_title": {
                "type": "string",
                "description": "Title of the plan (required for create)"
            },
            "plan_description": {
                "type": "string",
                "description": "Description of the plan (required for create)"
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of steps for the plan (required for create)"
            },
            "plan_id": {
                "type": "string",
                "description": "ID of the plan to operate on"
            },
            "step_index": {
                "type": "integer",
                "description": "Index of the step to mark (for mark_step command)"
            },
            "status": {
                "type": "string",
                "enum": ["completed", "failed", "in_progress"],
                "description": "Status to set for the step"
            }
        },
        "required": ["command"]
    }
    _plans: Dict[str, Plan] = PrivateAttr(default_factory=dict)
    _active_plan_id: Optional[str] = PrivateAttr(default=None)

    
    def __init__(self):
        super().__init__()
    
    async def execute(self, command: str, **kwargs) -> ToolResult:
        """Execute a planning command."""
        try:
            if command == "create":
                return await self._create_plan(**kwargs)
            elif command == "update":
                return await self._update_plan(**kwargs)
            elif command == "get":
                return await self._get_plan(**kwargs)
            elif command == "list":
                return await self._list_plans()
            elif command == "mark_step":
                return await self._mark_step(**kwargs)
            elif command == "delete":
                return await self._delete_plan(**kwargs)
            else:
                return ToolResult.error_result(f"Unknown command: {command}")
        except Exception as e:
            return ToolResult.error_result(f"Planning tool error: {str(e)}")
    
    async def _create_plan(self, plan_title: str, plan_description: str, steps: List[str], **kwargs) -> ToolResult:
        """Create a new plan."""
        plan = Plan(
            title=plan_title,
            description=plan_description,
            steps=[{"description": step, "status": "pending", "notes": ""} for step in steps]
        )
        
        self.plans[plan.id] = plan
        self.active_plan_id = plan.id
        
        return ToolResult.success_result({
            "plan_id": plan.id,
            "message": f"Created plan '{plan_title}' with {len(steps)} steps",
            "plan": plan.dict()
        })
    
    async def _get_plan(self, plan_id: Optional[str] = None, **kwargs) -> ToolResult:
        """Get a specific plan or the active plan."""
        target_id = plan_id or self.active_plan_id
        
        if not target_id:
            return ToolResult.error_result("No active plan and no plan_id provided")
        
        plan = self.plans.get(target_id)
        if not plan:
            return ToolResult.error_result(f"Plan {target_id} not found")
        
        return ToolResult.success_result(plan.dict())
    
    async def _list_plans(self, **kwargs) -> ToolResult:
        """List all plans."""
        plans_summary = []
        for plan_id, plan in self.plans.items():
            completed_steps = sum(1 for step in plan.steps if step["status"] == "completed")
            total_steps = len(plan.steps)
            
            plans_summary.append({
                "id": plan_id,
                "title": plan.title,
                "status": plan.status,
                "progress": f"{completed_steps}/{total_steps}",
                "active": plan_id == self.active_plan_id
            })
        
        return ToolResult.success_result(plans_summary)
    
    async def _mark_step(self, plan_id: Optional[str], step_index: int, status: str, **kwargs) -> ToolResult:
        """Mark a step with a specific status."""
        target_id = plan_id or self.active_plan_id
        
        if not target_id:
            return ToolResult.error_result("No active plan and no plan_id provided")
        
        plan = self.plans.get(target_id)
        if not plan:
            return ToolResult.error_result(f"Plan {target_id} not found")
        
        if step_index < 0 or step_index >= len(plan.steps):
            return ToolResult.error_result(f"Invalid step index: {step_index}")
        
        plan.steps[step_index]["status"] = status
        plan.current_step = step_index + 1 if status == "completed" else step_index
        
        return ToolResult.success_result({
            "message": f"Step {step_index} marked as {status}",
            "plan": plan.dict()
        })
    
    async def _delete_plan(self, plan_id: str, **kwargs) -> ToolResult:
        """Delete a plan."""
        if plan_id not in self.plans:
            return ToolResult.error_result(f"Plan {plan_id} not found")
        
        del self.plans[plan_id]
        
        if self.active_plan_id == plan_id:
            self.active_plan_id = None
        
        return ToolResult.success_result(f"Plan {plan_id} deleted")