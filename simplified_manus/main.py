import asyncio
from typing import Dict, Any
from llm import LLM, LLMConfig
from agents.planning import PlanningAgent
from agents.toolcall import ToolCallAgent
from tools.base import ToolCollection
from tools.planning import PlanningTool
from tools.web_search import WebSearchTool
from tools.browser_use import BrowserUseTool
from tools.terminate import TerminateTool
from dotenv import load_dotenv
import os

load_dotenv()

class SimplifiedOpenManus:
    """Simplified implementation of OpenManus."""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm = LLM(llm_config)
        self.setup_agents()
    
    def setup_agents(self):
        """Setup different types of agents."""
        # Planning Agent
        self.planning_agent = PlanningAgent(
            llm=self.llm,
            available_tools=ToolCollection(
                PlanningTool(),
                WebSearchTool(),
                BrowserUseTool(),
                TerminateTool()
            )
        )
        
        # General Tool Agent
        self.general_agent = ToolCallAgent(
            name="general_agent",
            description="General purpose agent with web capabilities",
            system_prompt="You are a helpful assistant that can search the web and use browser tools to help users.",
            llm=self.llm,
            available_tools=ToolCollection(
                WebSearchTool(),
                BrowserUseTool(),
                TerminateTool()
            )
        )
    
    async def run_planning_task(self, task: str) -> str:
        """Run a task using the planning agent."""
        print(f"Running planning task: {task}")
        result = await self.planning_agent.run(task)
        return result
    
    async def run_general_task(self, task: str) -> str:
        """Run a task using the general agent."""
        print(f"Running general task: {task}")
        result = await self.general_agent.run(task)
        return result
    

async def main():
    # Configure LLM (replace with your actual API key)
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key= os.getenv("OPENAI_API_KEY"),
        temperature=0.0
    )
    
    # Create OpenManus instance
    manus = SimplifiedOpenManus(llm_config)
    
    # Example 1: Planning task
    print("=== Planning Task Example ===")
    planning_task = "I need to research the latest trends in AI and create a comprehensive report. Help me plan this task."
    planning_result = await manus.run_planning_task(planning_task)
    print(f"Planning Result: {planning_result}")
    
    # Example 2: General task
    print("\n=== General Task Example ===")
    general_task = "Search for information about Python async programming and summarize the key concepts."
    general_result = await manus.run_general_task(general_task)
    print(f"General Result: {general_result}")

if __name__ == "__main__":
    asyncio.run(main())