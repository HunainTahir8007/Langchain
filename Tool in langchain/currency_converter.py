from langchain_core.tools import tool , InjectedToolArg
from typing import Annotated
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage


GROQ_API_KEY =""
llm = ChatGroq(api_key=GROQ_API_KEY , model = "llama-3.3-70b-versatile" , temperature=0.4)
API_KEY = ""
@tool 
def currency_converter( amount : float , base_currency : str , target_currency : str) -> float:
    """Converts an amount from one currency to another"""
    url =  f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    rate = data["conversion_rates"][target_currency]
    return rate * amount
    
  
llm_with_tools = llm.bind_tools([currency_converter])

messages = []
query = HumanMessage("convert the 100 USD to PKR")
messages.append(query)

result = llm_with_tools.invoke(messages)
messages.append(result)

tool_call = result.tool_calls[0]
tool_result = currency_converter.invoke(
    tool_call["args"]
)
messages.append(
    ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"]
    ))

final = llm_with_tools.invoke(messages)
print(final.content)
