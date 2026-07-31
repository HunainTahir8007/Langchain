from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate 
from langchain_core.tools import tool 
from langchain_groq import ChatGroq 
from datetime import datetime


llm = ChatGroq(api_key="" , model="llama-3.3-70b-versatile" , temperature=0.3)

@tool
def addition(a:int , b:int ) -> int:
    """this function returns the sum of the a and b"""
    return (a+b)

@tool
def multiply(a:int  , b: int) -> int:
    """this function returnsthe product of the a and b"""
    return a*b

@tool
def timedate()->str:
    """this function returns the current time"""
    return str(datetime.now())


@tool
def word_count(text : str) ->int:
    """count number of words in text"""
    return int(len(text.split()))

tools = [addition , multiply , timedate , word_count]

my_agent=create_agent(
    model=llm ,
    tools=tools , 
    system_prompt= """
You are a senior AI assistant.

Analyze every question carefully.

Use tools whenever required.

For mathematical questions,
always use the math tools.

For date or time,
use the timedate tool.

Always provide clear and helpful answers.
""" 
)


res = my_agent.invoke(
    {
        "messages" : [
            {
                "role":  "user" , 
                "content" :"can u tell me the current time if yes then tell me and also count words in this query ?"
            }
        ]
    }
)
print(res["messages"][-1].content)