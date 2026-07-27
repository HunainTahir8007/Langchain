from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch ,RunnableLambda # for conditional runnig 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal 
import sys

sys.stdout.reconfigure(encoding="utf-8")
class sentiment_analysis(BaseModel):
     sentiment : Literal["positive" , "negative"] = Field(description="Give the sentiment of the feedback")
     
     
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite") 

parser1 = PydanticOutputParser(pydantic_object=sentiment_analysis)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Classfity the sentiment of the following textt into the positive or negative \n {text} \n{format_instruction}", 
    input_variables=["text"],
    partial_variables= {"format_instruction": parser1.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write an appropiate  response for this positive feedback just in one line \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Write an appropiate  response for this negative feedback just in one line \n {text}",
    input_variables=['text']
)
classifier_chain = prompt1 | model | parser1

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=="positive" , prompt2 | model | parser),
    (lambda x:x.sentiment=="negative" , prompt3 | model | parser),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({"text": "what a day "})
print(result)
