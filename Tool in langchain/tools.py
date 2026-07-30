from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

parser  =  StrOutputParser()

llm  = ChatGroq(api_key= , model="llama-3.3-70b-versatile" , temperature=0.4)

messages = []

query = HumanMessage("provide me the sum of the 3 and 9")
messages.append(query)
@tool
def multiply(a: int , b :int ) -> int : 
    """This function recieives 2 integers and return its product"""
    return a*b

@tool
def addition(a: int , b :int ) -> int : 
    """This function recieives 2 integers and return its sum"""
    return a+b


llm_with_tools = llm.bind_tools([multiply, addition])
result = llm_with_tools.invoke(messages)
messages.append(result)
tool_call = result.tool_calls[0]
tool_result = addition.invoke(result.tool_calls[0])
messages.append(tool_result)

final = llm_with_tools.invoke(messages)
print(final.content)