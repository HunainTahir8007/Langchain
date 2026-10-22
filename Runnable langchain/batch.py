from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnableLambda , RunnablePassthrough , RunnableBranch
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
    template="Generate me deatialed explaination  on the topic :- \n{text}" , 
    input_variables=['text']
    
)

prompt2 =  PromptTemplate(
    template="Summerize this text :- {text} ", 
    input_variables=['text']

)


explaination_chain = RunnableSequence(prompt1 , model , parser)

conditional_chain = RunnableBranch(
    (lambda x : len(x.split())>10, RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough()
)


final_chain = explaination_chain | conditional_chain

result = final_chain.invoke({'text' : "AI"})
print(result)
