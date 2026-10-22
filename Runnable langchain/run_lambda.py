from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnableLambda , RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from typing import  TypedDict,Annotated 
import sys

sys.stdout.reconfigure(encoding="utf-8")


load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite" , temperature = 0.8)

parser=StrOutputParser()


prompt1 = PromptTemplate(
    template="Generate me the onlt one  joke on the topic :- \n{text}" , 
    input_variables=['text']
    
)

joke_chain = prompt1 | model | parser

parallel_chain = RunnableParallel(
    {
        "joke" : RunnablePassthrough() , 
        "len": RunnableLambda(lambda x : len(x.split()))
        
    }
    
)
chain = joke_chain | parallel_chain

response =chain.invoke({"text": "AI"})
print(response['joke'])
print(response['len'])