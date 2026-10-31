from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite" , temperature = 0.8)

class my(BaseModel):
    name: str = Field("fetch the name from the text")
    age : int = Field("fetch the age from the text")


parsor = PydanticOutputParser(pydantic_object=my)

prompt = PromptTemplate(
    template="your duty is to fetch the name and age from the \n {text} \n {format_instruction}" ,
    input_variables= ['text'],
    partial_variables={
        "format_instruction": parsor.get_format_instructions()
    }
    
)

chain = RunnableSequence(prompt , model , parsor)
result = chain.invoke({"text": "My name is hunain tahir and my is is 18"})

print(result)