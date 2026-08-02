from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema   
from pydantic import BaseModel , Field
load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite" , temperature = 0.7)
class person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int =Field(gt=18 , description="age of the  person ")
    city : str = Field(description="Name of the city wo which the person belongs ")
    
    
parsar = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template="give me the name , age  and city of the finctional {place} person {format_instruction}" ,
    input_variables=['place'],
     partial_variables={
        "format_instruction": parsar.get_format_instructions()
    }
    
)

chain = template | model | parsar
finl = chain.invoke({"place":"indian"})
print(finl)