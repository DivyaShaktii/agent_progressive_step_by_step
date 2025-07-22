import asyncio
import os
from typing import Dict, List, Optional, Annotated, Sequence
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel
import sys
from io import StringIO

load_dotenv()

class AgentState(BaseModel):
    """State for the agent graph"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Initialize the LLM
llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# Define tools using LangChain's tool decorator
@tool
def python_executor(code: str) -> str:
    """Execute Python code and return the results"""
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Create namespace
        namespace = {}
        
        # Execute code
        exec(code, namespace)
        
        # Get output
        output = captured_output.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        if not output:
            try:
                result = eval(code, namespace)
                output = str(result)
            except:
                output = "Code executed successfully (no output)"
        
        return output.strip()
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {str(e)}"

# Initialize Tavily search tool
tavily_search = TavilySearchResults(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY")
)

# Create tool list
tools = [python_executor, tavily_search]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Create tool node
tool_node = ToolNode(tools)

class LangGraphAgent:
    def __init__(self):
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
    
    def _build_graph(self) -> StateGraph:
        """Build the agent graph with thinking and acting nodes"""
        
        def should_continue(state: AgentState) -> str:
            """Determine if we should continue to tools or end"""
            messages = state.messages
            last_message = messages[-1]
            
            # If there are tool calls, continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            else:
                return "end"
        
        def call_model(state: AgentState) -> Dict:
            """Call the model with the current state"""
            messages = state.messages
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        
        return workflow
    
    async def run(self, user_input: str, thread_id: str = "default") -> str:
        """Run the agent with user input"""
        config = RunnableConfig(
            configurable={"thread_id": thread_id}
        )
        
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        # Get the last AI message from the result
        if "messages" in result:
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage) and message.content:
                    return message.content
        
        return "No response generated"

    async def run_with_streaming(self, user_input: str, thread_id: str = "default"):
        """Run the agent with streaming output for better user experience"""
        config = RunnableConfig(
            configurable={"thread_id": thread_id}
        )
        
        print(f"User: {user_input}")
        print("Agent is thinking...")
        
        async for event in self.app.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values"
        ):
            if "messages" in event:
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage):
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        print(f"Agent is using tools: {[tool_call['name'] for tool_call in last_message.tool_calls]}")
                    elif last_message.content:
                        print(f"Agent: {last_message.content}")
                        return last_message.content


# Usage example
async def main():
    agent = LangGraphAgent()
    
    # print("Starting LangGraph Agent...")
    # response = await agent.run(
    #     "Calculate the square root of 1876982627.8 and find the mathematician who invented square root"
    # )
    # print(f"Agent Response: {response}")
    print("\n" + "="*50 + "\n")

    await agent.run_with_streaming(
        #"What is 15 * 23 + 42? Also search for information about the Fibonacci sequence."
        "Calculate the square root of 1876982627.8 and find the mathematician who invented square root"
    )

if __name__ == "__main__":
    asyncio.run(main())