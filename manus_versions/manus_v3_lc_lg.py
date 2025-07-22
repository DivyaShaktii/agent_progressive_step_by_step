import os
import sys
from typing import Annotated, List

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                        ToolMessage)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor , ToolNode
from typing_extensions import TypedDict
from termcolor import colored
from io import StringIO
from langchain.chat_models import init_chat_model
import asyncio

# --- 1. Define Tools ---
# We use the simple @tool decorator from LangChain. This replaces your
# BaseTool class and the complex parameter definitions.

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

# --- 2. Define the Agent's State ---
# This is a simple dictionary that holds the state of our agent.
# LangGraph will automatically manage this state for us.
# The `add_messages` function ensures new messages are appended to the list.

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]


# --- 3. Define the Graph Nodes ---
# These are the functions that represent the steps in our ReAct loop.

async def call_model_node(state: AgentState, llm_with_tools):
    """
    The "Think" step. This node invokes the LLM with the current message
    history and the available tools. The LLM's response (either a direct
    answer or tool calls) is added to the state.
    """
    print(colored("\n🤔 Thinking...", "yellow"))
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def call_tool_node(state: AgentState, tool_executor):
    """
    The "Act" step. This node checks the last message for tool calls.
    If it finds any, it executes them and returns the results as
    ToolMessage objects.
    """
    print(colored("🎬 Acting...", "magenta"))
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    # The ToolExecutor from LangGraph handles executing the tools and
    # formatting the output for us.
    response = await tool_executor.abatch(tool_calls)
    
    # We convert the responses into ToolMessage objects
    tool_messages = [
        ToolMessage(content=str(res), tool_call_id=call["id"])
        for res, call in zip(response, tool_calls)
    ]
    
    return {"messages": tool_messages}


# --- 4. Define the Conditional Edge ---
# This function decides the next step after the LLM has been called.

def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue with a tool call or end the process.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print(colored("✅ Decision: Use tools.", "yellow"))
        return "continue"
    else:
        print(colored("✅ Decision: No tools needed. Responding directly.", "yellow"))
        return "end"


# --- 5. Assemble the Graph ---
# Here, we wire all our components together into a state machine.

# Initialize the LLM


def build_graph():
    """
    Builds and compiles the LangGraph agent.
    """
    # Initialize the LLM and bind the tools to it. This tells the LLM
    # what functions it can call. This replaces your manual system prompt.
    llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)

    # Agent Node: This is the "think" step. It calls the LLM.
    # We define it here so it has access to llm_with_tools from the parent scope.
    async def call_model_node(state: AgentState):
        print(colored("\n🤔 Thinking...", "yellow"))
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    # Action Node: This is the "act" step. We use the pre-built ToolNode.
    # This replaces the old ToolExecutor and the custom call_tool_node function.
    tool_node = ToolNode(tools)


    # Define the StateGraph with our custom AgentState
    graph = StateGraph(AgentState)

    # Add the nodes to the graph
    graph.add_node("agent", call_model_node)
    graph.add_node("action", tool_node)

    # Define the edges that connect the nodes
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "action", "end": END},
    )
    graph.add_edge("action", "agent")

    # Compile the graph into a runnable application
    return graph.compile()


# --- 6. Run the Agent ---

async def main():
    """
    Sets up and runs the agent with a user query, streaming the final output.
    """
    agent_app = build_graph()
    query = "What is the square root of 1876982627.8, and who is considered the first mathematician to have formally described the concept of irrational numbers resulting from square roots?"
    
    print(colored("🚀 Starting ReAct Agent...", "white", "on_blue"))
    print(colored(f"\n>>>>> User Input: {query}", "blue", attrs=['bold']))

    # The `astream` method handles the entire ReAct loop and streams the
    # final output token by token.
    final_response_printed = False
    async for event in agent_app.astream({"messages": [HumanMessage(content=query)]}):
        # We are only interested in the final answer from the 'agent' node
        if "agent" in event:
            message = event["agent"]["messages"][-1]
            if isinstance(message, AIMessage) and not message.tool_calls:
                if not final_response_printed:
                    print(colored("\n✅ Final Response:", "cyan", attrs=['bold']))
                    final_response_printed = True
                # Print the streaming content
                print(colored(message.content, "cyan"), end="", flush=True)

    print(colored("\n\n🏁 Agent run finished.", "white", "on_blue"))


if __name__ == "__main__":
    asyncio.run(main())