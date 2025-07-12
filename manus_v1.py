from typing import List, Optional
from pydantic import BaseModel , Field, ConfigDict
import asyncio
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

class SimpleMessage(BaseModel):
    role : str
    content : str

class LLM :
    def __init__(self, model_provider = "google_genai",
                 model_name = "gemini-2.5-flash"):
        self.llm = init_chat_model(model = model_name,model_provider=model_provider)

    async def ask(self, messages : List[SimpleMessage]) -> str:
        formatted_messages = [
        {"role": msg.role, "content": msg.content} 
        for msg in messages ]
        response = await self.llm.ainvoke(formatted_messages)
        return response.content


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

async def main():
    agent = SimpleAgent(llm = LLM())
    response = await agent.run("Hello, how are you ? ")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())





