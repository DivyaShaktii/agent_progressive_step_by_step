from typing import Dict, Any, Optional
from agents.toolcall import ToolCallAgent
from tools.base import ToolCollection
from tools.planning import PlanningTool
from tools.terminate import TerminateTool
from schema import Message
from pydantic import Field
from schema import Message, ToolCall

class PlanningAgent(ToolCallAgent):
    """Agent that creates and manages plans to solve tasks."""
    
    name: str = "planning_agent"
    description: str = "An agent that creates and manages plans to solve tasks"
    system_prompt: str = """You are a planning agent that excels at breaking down complex tasks into manageable steps.
    
Your responsibilities:
1. Analyze the user's request to understand the task scope
2. Create detailed, actionable plans using the planning tool
3. Execute steps systematically
4. Track progress and adapt plans as needed
5. Use terminate when the task is complete

Always think step by step and create comprehensive plans before executing actions."""
    
    available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(
        PlanningTool(), TerminateTool()
    ))
    
    active_plan_id: Optional[str] = None
    
    async def think(self) -> bool:
        """Think with current plan context."""
        messages = self.memory.get_messages()
        
        # Add plan context if we have an active plan
        if self.active_plan_id:
            planning_tool = next(tool for tool in self.available_tools.tools if tool.name == "planning")
            plan_result = await planning_tool.execute("get", plan_id=self.active_plan_id)
            
            if plan_result.success:
                plan_info = plan_result.output
                plan_context = f"""
CURRENT PLAN STATUS:
Title: {plan_info['title']}
Description: {plan_info['description']}
Current Step: {plan_info['current_step']} / {len(plan_info['steps'])}
Steps:
"""
                for i, step in enumerate(plan_info['steps']):
                    status_indicator = "✓" if step['status'] == 'completed' else "→" if i == plan_info['current_step'] else "○"
                    plan_context += f"{status_indicator} {i+1}. {step['description']} ({step['status']})\n"
                
                messages.append(Message.user_message(plan_context))
        
        # Add system prompt
        if self.system_prompt:
            messages = [Message.system_message(self.system_prompt)] + messages
        
        # Get LLM response
        response = await self.llm.ask_tool(
            messages=messages,
            tools=self.available_tools.to_params()
        )
        
        # Update memory
        #self.update_memory("assistant", response.get("content", ""))
        self.update_memory("assistant", response.get("content") or "")
        
        # # Extract tool calls
        # tool_calls = response.get("tool_calls", [])
        # self.tool_calls = [ToolCall(**call) for call in tool_calls] if tool_calls else []
        tool_calls = response.get("tool_calls", [])
        self.tool_calls = [ToolCall(**(call if isinstance(call, dict) else call.__dict__))  
                           for call in tool_calls ] if tool_calls else []


        return len(self.tool_calls) > 0
    
    async def act(self) -> str:
        """Execute tools and update plan state."""
        result = await super().act()
        
        # Update active plan ID if a plan was created
        for tool_call in self.tool_calls:
            if tool_call.function["name"] == "planning":
                args = tool_call.function.get("arguments", {})
                if isinstance(args, str):
                    import json
                    args = json.loads(args)
                
                if args.get("command") == "create":
                    # Extract plan ID from the result
                    planning_tool = next(tool for tool in self.available_tools.tools if tool.name == "planning")
                    self.active_plan_id = planning_tool.active_plan_id
        
        return result