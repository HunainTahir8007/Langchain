from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite" , temperature = 0.7)

parsor =JsonOutputParser()

tempelate = PromptTemplate(
    template="Give me 5 facts about {topic}  {format_instruction}" , 
    input_variables=["topic"],
    partial_variables={"format_instruction" : parsor.get_format_instructions()}
        
)
chain = tempelate | model | parsor
result = chain.invoke({"topic": "black hole"})
print(result)