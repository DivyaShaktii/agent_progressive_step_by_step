from abc import abstractmethod
from agents.base import BaseAgent
from schema import AgentState

class ReActAgent(BaseAgent):
    """Agent implementing the ReAct (Reason and Act) pattern."""
    
    @abstractmethod
    async def think(self) -> bool:
        """Think about the current situation and decide if action is needed."""
        pass
    
    @abstractmethod
    async def act(self) -> str:
        """Execute the decided action."""
        pass
    
    async def step(self) -> str:
        """Execute one ReAct step: think, then act."""
        self.state = AgentState.THINKING
        
        # Think phase
        should_act = await self.think()
        
        if not should_act:
            self.state = AgentState.COMPLETED
            return "Thinking complete - no action needed"
        
        # Act phase
        self.state = AgentState.ACTING
        result = await self.act()
        
        return result