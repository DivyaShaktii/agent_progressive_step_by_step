from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
import asyncio
import json
import sys , os
from io import StringIO
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import httpx

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

class LLM :
    def __init__(self, model_provider = "google_genai",
                 model_name = "gemini-2.5-flash"):
        self.llm = init_chat_model(model = model_name,model_provider=model_provider)

    async def ask(self, messages : List[SimpleMessage]) -> str:
        formatted_messages = []
        for msg in messages:
            if msg.role == "user":
                formatted_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                formatted_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                formatted_messages.append(SystemMessage(content=msg.content))
        
        print(colored(">>> Calling LLM without tools...", "green"))
        response = await self.llm.ainvoke(formatted_messages)
        print(colored(f"<<< LLM Response: {response.content}", "green"))
        return response.content
    
    async def ask_with_tools(self, messages: List[SimpleMessage], tools: List[Dict]) -> Dict:
        """Ask LLM with tool calling capability"""
        formatted_messages = []
        for msg in messages:
            if msg.role == "user":
                formatted_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                formatted_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                formatted_messages.append(SystemMessage(content=msg.content))
        
        # Add system message about available tools
        tool_descriptions = []
        for tool in tools:
            func_info = tool["function"]
            tool_descriptions.append(f"- {func_info['name']}: {func_info['description']}")
        
        system_msg = SystemMessage(content=f"""You are an AI assistant with access to the following tools:
                    {chr(10).join(tool_descriptions)}

                    When you need to use a tool, respond with a JSON object in this format:
                    {{"tool_calls": [{{"function": {{"name": "tool_name", "arguments": {{"param": "value"}}}}}}]}}

                    If you don't need to use any tools, just respond normally.""")
                            
        formatted_messages.insert(0, system_msg)
        
        print(colored(">>> Calling LLM with tools...", "cyan"))
        response = await self.llm.ainvoke(formatted_messages)
        print(colored(f"<<< LLM Raw Response with tools: {response.content}", "cyan"))
        
        # Try to parse tool calls from response
        content = response.content.strip()
        try:
            # Check if response contains tool calls
            if content.startswith('{') and 'tool_calls' in content:
                parsed = json.loads(content)
                return parsed
            else:
                return {"content": content, "tool_calls": []}
        except json.JSONDecodeError:
            return {"content": content, "tool_calls": []}


class SimpleAgent(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed = True)
    name : str = "SimpleAgent"
    llm  : LLM 
    messages : List[SimpleMessage] = Field(default_factory= list)

    # class Config:
    #     arbitrary_types_allowed = True

    async def run(self ,  user_input : str) -> str :
        self.messages.append(SimpleMessage(role = "user", content = user_input))
        response = await self.llm.ask(self.messages)

        self.messages.append(SimpleMessage(role= "assistant" , content= response))  
        return response

class ToolResult(BaseModel):
    output : Any = None
    error : Optional[str]
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
                "parameters": self.parameters or {}
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
            print(colored(f"   Tool calls: {json.dumps(self.tool_calls, indent=2)}", "yellow"))
        else:
            print(colored("❌ Decision: No tools needed. Responding directly.", "yellow"))
        return len(self.tool_calls)> 0
    
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
                results.append(f"Tool {tool_name} not found")
        
        tool_outputs = "\n".join(results)
        print(colored(f"🎬 Tool execution finished. Combined output:\n{tool_outputs}", "magenta"))
        return "\n".join(results)
    
    async def run(self, user_input : str) -> str :
        print(colored(f"\n>>>>> User Input: {user_input}", "blue", attrs=['bold']))
        self.messages.append(SimpleMessage(role = "user", content= user_input))

        should_act = await self.think()

        if should_act:
            tool_results = await self.act()

            self.messages.append(SimpleMessage(role = "assistant", content= tool_results))
            
            print(colored("\n🧠 Synthesizing final response based on tool results...", "yellow"))
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
        

async def main():
    agent = ToolCallAgent(llm= LLM())
    agent.available_tools.add_tool(PythonExecuteTool())
    agent.available_tools.add_tool(TavilySearchTool())
    print(colored("🚀 Starting agent...", "white", "on_blue"))
    response = await agent.run("Calculate the square root of 1876982627.8 and find the mathmetician who invented square root")
    print(colored("\n🏁 Agent run finished.", "white", "on_blue"))

if __name__ == "__main__" :
    asyncio.run(main())
