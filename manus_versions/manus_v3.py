from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field, ConfigDict
from io import StringIO
import asyncio
import json
import sys , os , httpx
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


from dotenv import load_dotenv

load_dotenv()

try:
    from termcolor import colored
except ImportError:
    print("Warning: 'termcolor' library not found. For colored output, please install it using 'pip install termcolor'")
    def colored(text, *args, **kwargs):
        return text

class SimpleMessage(BaseModel):
    role : str
    content : str


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    FINISHED = "finished"
    STUCK = "stuck"

class LLM :
    def __init__(self, model_provider = "google_genai",
                 model_name = "gemini-2.5-flash"):
        self.llm = init_chat_model(model = model_name,model_provider=model_provider)

    def _format_messages(self, messages: List[SimpleMessage]) -> List:
        """Convert SimpleMessage objects to LangChain message format"""
        formatted_messages = []
        for msg in messages:
            if msg.role == "user":
                formatted_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                formatted_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                formatted_messages.append(SystemMessage(content=msg.content))
        return formatted_messages

    async def ask(self, messages : List[SimpleMessage]) -> str:
        formatted_messages = self._format_messages(messages)
        print(colored(">>> Calling LLM...", "green"))
        response = await self.llm.ainvoke(formatted_messages)
        print(colored(f"<<< LLM Response: {response.content[:100]}...", "green"))
        return response.content
    
    async def ask_stream(self, messages: List[SimpleMessage]) -> AsyncGenerator[str, None]:
        """Stream LLM response"""
        formatted_messages = self._format_messages(messages)
        print(colored(">>> Streaming LLM response...", "green"))
        
        async for chunk in self.llm.astream(formatted_messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    async def ask_with_tools(self, messages: List[SimpleMessage], tools: List[Dict]) -> Dict:
        """Ask LLM with tool calling capability"""
        formatted_messages = []
        formatted_messages = self._format_messages(messages)
        
        # Add system message about available tools
        tool_descriptions = [f"- {tool['function']['name']}: {tool['function']['description']}" 
                           for tool in tools]
        
        system_msg = SystemMessage(content=f"""You are an AI assistant with access to the following tools:
{chr(10).join(tool_descriptions)}

When you need to use a tool, respond with a JSON object in this format:
{{
    "tool_calls": [
        {{
            "function": {{
                "name": "tool_name",
                "arguments": {{
                    "param": "value"
                }}
            }}
        }}
    ]
}}

If you don't need to use any tools, just respond normally.""")
                            
        formatted_messages.insert(0, system_msg)
        
        print(colored(">>> Calling LLM with tools...", "cyan"))
        response = await self.llm.ainvoke(formatted_messages)
        print(colored(f"<<< LLM Response with tools: {response.content[:100]}...", "cyan"))
        
        # Try to parse tool calls from response
        content = response.content.strip()
        if content.startswith('{') and 'tool_calls' in content:
            try:
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError:
                return {"content": content, "tool_calls": []}
        else:
            return {"content": content, "tool_calls": []}


class ToolResult(BaseModel):
    output : Any = None
    error : Optional[str] = None
    success : bool = True
    
class BaseTool(ABC, BaseModel):
    name : str
    description: str
    parameters : Optional[Dict] = None

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult :
        """Execute the tool with given parameters"""
        pass

    def to_param(self) -> Dict :
        """Convert tool to OpenAI function format"""
        return {
            "type" : "function",
            "function" : {
                "name" : self.name,
                "description" : self.description,
                "parameters": self.parameters or {
                     "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

class PythonExecuteTool(BaseTool):
    name : str = "python_execute"
    description : str = "Execute Python code and return results"
    parameters : Dict = {
        "type" : "object" ,
        "properties" : {
            "code" : {
                "type" : "string",
                "description" : "Python code to execute"
            }
        },
        "required" : ["code"]
    }

    async def execute(self, code: str) -> ToolResult:
        try:
            # Capture stdout for print statements
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Create a namespace for execution
            namespace = {}
            
            # Execute the code
            exec(code, namespace)
            
            # Get the captured output
            output = captured_output.getvalue()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            if not output:
                # Try to evaluate as an expression
                try:
                    result = eval(code, namespace)
                    output = str(result)
                except:
                    output = "Code executed successfully (no output)"
            
            return ToolResult(output=output.strip(), success=True)
        except Exception as e:
            # Restore stdout in case of error
            sys.stdout = old_stdout
            return ToolResult(error=str(e), success=False)

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = "Search the web using Tavily API for current information"
    api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    base_url: str = "https://api.tavily.com/search"
    parameters: Dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 3)",
                "default": 3
            }
        },
        "required": ["query"]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")

    async def execute(self, query: str, max_results: int = 3) -> ToolResult:
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract the answer if available
                    answer = data.get("answer", "")
                    results = data.get("results", [])
                    
                    # Format the results
                    formatted_results = []
                    if answer:
                        formatted_results.append(f"Answer: {answer}")
                    
                    if results:
                        formatted_results.append("\nDetailed Results:")
                        for i, result in enumerate(results[:max_results], 1):
                            title = result.get("title", "No title")
                            url = result.get("url", "")
                            content = result.get("content", "No content available")
                            
                            formatted_results.append(f"{i}. {title}")
                            if url:
                                formatted_results.append(f"   URL: {url}")
                            formatted_results.append(f"   Content: {content[:300]}...")
                            formatted_results.append("")
                    
                    final_output = "\n".join(formatted_results)
                    
                    return ToolResult(output=final_output, success=True)
                else:
                    error_msg = f"Tavily API error: {response.status_code} - {response.text}"
                    return ToolResult(error=error_msg, success=False)
                    
        except httpx.TimeoutException:
            return ToolResult(error="Search request timed out", success=False)
        except httpx.RequestError as e:
            return ToolResult(error=f"Network error: {str(e)}", success=False)
        except Exception as e:
            return ToolResult(error=f"Search failed: {str(e)}", success=False)

class ToolCollection(BaseModel):
    tools : List[BaseTool] = Field(default_factory = list)

    def add_tool(self, tool : BaseTool):
        self.tools.append(tool)

    def get_tool(self , name : str) -> Optional[BaseTool]:
        return next((tool for tool in self.tools if tool.name == name), None)
    
    def to_params(self) -> List[Dict]:
        return [tool.to_param() for tool in self.tools]
    
class SimpleAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed= True)

    name : str = "SimpleAgent"
    llm : LLM
    messages: List[SimpleMessage] = Field(default_factory= list)

    async def run(self, user_input: str) -> str:
        self.messages.append(SimpleMessage(role="user", content=user_input))
        response = await self.llm.ask(self.messages)
        self.messages.append(SimpleMessage(role="assistant", content=response))  
        return response

class ToolCallAgent(SimpleAgent):
    available_tools : ToolCollection = Field(default_factory= ToolCollection)
    tool_calls : List[Dict] = Field(default_factory= list)

    async def think(self) -> bool :
        """Analyze the situation and decide which tools to use"""
        print(colored("\n🤔 Thinking... Analyzing user input to decide on tool usage.", "yellow"))
        response = await self.llm.ask_with_tools(
            messages = self.messages,
            tools = self.available_tools.to_params())
        self.tool_calls = response.get("tool_calls",[])
        if self.tool_calls:
            print(colored(f"✅ Decision: Use tools. Found {len(self.tool_calls)} tool calls.", "yellow"))
            for i, tool_call in enumerate(self.tool_calls):
                print(colored(f"   Tool {i+1}: {tool_call['function']['name']}", "yellow"))
        else:
            print(colored("❌ Decision: No tools needed. Responding directly.", "yellow"))
        
        return len(self.tool_calls) > 0
    
    async def act(self) -> str :
        """Execute the selected tools"""
        print(colored("\n🎬 Acting... Executing selected tools.", "magenta"))
        results = []

        for tool_call in self.tool_calls :
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            print(colored(f"   - Using tool: `{tool_name}` with arguments: {tool_args}", "magenta"))

            tool = self.available_tools.get_tool(tool_name)
            if tool:
                result = await tool.execute(**tool_args)
                if result.success:
                    print(colored(f"     ✅ Tool `{tool_name}` executed successfully.", "green"))
                    print(colored(f"     Output: {result.output}", "green"))
                    results.append(f"Tool {tool_name} returned: {result.output}")
                else:
                    print(colored(f"     ❌ Tool `{tool_name}` failed. Error: {result.error}", "red"))
                    results.append(f"Tool {tool_name} failed with error: {result.error}")
            else:
                error_msg = f"Tool {tool_name} not found"
                print(colored(f"     ❌ {error_msg}", "red"))
                results.append(error_msg)
        
        return "\n".join(results)
    
    async def run(self, user_input : str) -> str :
        print(colored(f"\n>>>>> User Input: {user_input}", "blue", attrs=['bold']))
        self.messages.append(SimpleMessage(role = "user", content= user_input))

        should_act = await self.think()

        if should_act:
            tool_results = await self.act()

            self.messages.append(SimpleMessage(role = "assistant", content= tool_results))
            
            print(colored("\n🧠 Synthesizing final response ...", "yellow"))
            final_response = await self.llm.ask(self.messages)
            
            self.messages.append(SimpleMessage(role = "assistant", content= final_response))
            print(colored(f"\n✅ Final Response:\n{final_response}", "cyan", attrs=['bold']))
            return final_response
        else :
            print(colored("\n💬 Generating direct response...", "yellow"))
            response = await self.llm.ask(self.messages)
            self.messages.append(SimpleMessage(role = "assistant", content = response))
            print(colored(f"\n✅ Final Response:\n{response}", "cyan", attrs=['bold']))
            return response

    async def run_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Stream the agent's response"""
        print(colored(f"\n>>>>> User Input: {user_input}", "blue", attrs=['bold']))
        self.messages.append(SimpleMessage(role="user", content=user_input))

        should_act = await self.think()

        if should_act:
            tool_results = await self.act()
            self.messages.append(SimpleMessage(role="assistant", content=tool_results))
            
            print(colored("\n🧠 Synthesizing final response...", "yellow"))
            
            full_response = ""
            async for chunk in self.llm.ask_stream(self.messages):
                full_response += chunk
                yield chunk
            
            self.messages.append(SimpleMessage(role="assistant", content=full_response))
        else:
            print(colored("\n💬 Generating direct response...", "yellow"))
            
            full_response = ""
            async for chunk in self.llm.ask_stream(self.messages):
                full_response += chunk
                yield chunk
            
            self.messages.append(SimpleMessage(role="assistant", content=full_response))

class ReActAgent(ToolCallAgent):
    state : AgentState = AgentState.IDLE
    max_steps : int = 6
    current_step : int = 0
    system_prompt : str = "You are a helpful AI assistant that thinks step by step."

    def format_conversation(self) -> str:
        """Format conversation history for context"""
        return "\n".join([f"{msg.role}: {msg.content}" for msg in self.messages[-3:]])  # Last 3 messages

    async def step(self) -> str: 
        """Execute one think-act cycle"""
        self.current_step += 1
        print(colored(f"\n--- Step {self.current_step} ---", "white", "on_blue"))

        if self.current_step > self.max_steps:
            self.state = AgentState.STUCK
            return "Maximum steps reached. Providing best available answer."
        
        # Think phase
        self.state = AgentState.THINKING
        should_continue = await self.think()
        
        if not should_continue:
            self.state = AgentState.FINISHED
            #return await self.finalize()
            return "Ready to provide final answer."
        
        # Act phase
        self.state = AgentState.ACTING
        action_result = await self.act()

       # Observe results and plan next action
        observation = await self.observe(action_result)
        
        return observation
    
    async def think(self) -> bool :
        """Enhanced thinking with step-by-step reasoning"""
        print(colored(f"\n🤔 ReAct Thinking... Step {self.current_step}", "yellow"))
        thinking_prompt = f"""
        Step {self.current_step}: Analyze the current situation and decide what to do next.
        
        Current conversation:
        {self.format_conversation()}
        
        Available tools: {[tool.name for tool in self.available_tools.tools]}
        
        Think step by step:
        1. What is the current goal?
        2. What information do I have?
        3. What do I need to do next?
        4. Should I use any tools?
        5. Or am I ready to provide a final answer?

        If you need to use tools, respond with the appropriate tool calls in JSON format.
        If you're ready to provide a final answer, respond with your reasoning only.
        """
        
        self.messages.append(SimpleMessage(role="user", content=thinking_prompt))
        
        response = await self.llm.ask_with_tools(
            messages=self.messages,
            tools=self.available_tools.to_params()
        )
        
        self.tool_calls = response.get("tool_calls", [])
        thinking_content = response.get("content", "")
        
        self.messages.append(SimpleMessage(role="assistant", content=thinking_content))
        
        if self.tool_calls:
            print(colored(f"✅ ReAct Decision: Use {len(self.tool_calls)} tool(s)", "yellow"))
            for i, tool_call in enumerate(self.tool_calls):
                print(colored(f"   Tool {i+1}: {tool_call['function']['name']}", "yellow"))
        else:
            print(colored("❌ ReAct Decision: No tools needed, ready for final answer", "yellow"))
        
        return len(self.tool_calls) > 0
    
    async def observe(self, action_result: str) -> str:
        """Observe the results and plan next steps"""
        print(colored(f"\n👁️  Observing results from Step {self.current_step}...", "blue"))
        observation_prompt = f"""
        Action results from Step {self.current_step}: {action_result}
        
        Observe and reflect:
        1. Did the action succeed?
        2. What did I learn?
        3. Am I closer to the goal?
        4. What should I do next?
        5. Do I have enough information to provide a final answer?
        
        Provide your observations and next steps.
        """
        
        self.messages.append(SimpleMessage(role="user", content=observation_prompt))
        
        response = await self.llm.ask(self.messages)
        self.messages.append(SimpleMessage(role="assistant", content=response))
        
        return response
    
    async def run(self, user_input: str) -> str:
        """Run the complete ReAct loop"""
        print(colored(f"\n>>>>> Starting ReAct Agent for: {user_input}", "blue", attrs=['bold']))
        self.messages.append(SimpleMessage(role="user", content=user_input))
        self.state = AgentState.IDLE
        self.current_step = 0
        
        while self.state not in [AgentState.FINISHED, AgentState.STUCK] and self.current_step < self.max_steps :
            step_result = await self.step()
            print(colored(f"Step {self.current_step} result: {step_result[:100]}...", "white"))
        
        return await self.finalize()
    
    async def finalize(self) -> str:
        """Provide final answer"""
        print(colored("\n🎯 Finalizing response...", "green"))
        final_prompt = "Based on our conversation, provide a comprehensive final answer."
        self.messages.append(SimpleMessage(role="user", content=final_prompt))
        
        response = await self.llm.ask(self.messages)
        print(colored(f"\n✅ Final Answer:\n{response}", "cyan", attrs=['bold']))
        return response
    

async def main():
    agent = ReActAgent(llm= LLM())

    # Add tools :
    agent.available_tools.add_tool(PythonExecuteTool())
    agent.available_tools.add_tool(TavilySearchTool())

    print(colored("🚀 Starting ReAct Agent...", "white", "on_blue"))
    
    # Example with streaming
    print(colored("\n--- Streaming Example ---", "magenta"))
    async for chunk in agent.run_stream("What is the square root of 16 and who invented the square root concept?"):
        print(chunk, end='', flush=True)
    
    print(colored("\n\n--- ReAct Example ---", "magenta"))
    response = await agent.run("What is the square root of 1876982627.8, and who is considered the first mathematician to have formally described the concept of irrational numbers resulting from square roots?")
    
    print(colored("\n🏁 Agent run finished.", "white", "on_blue"))
if __name__ == "__main__" :
    asyncio.run(main())