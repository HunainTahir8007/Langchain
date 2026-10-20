from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite") 

parser = StrOutputParser()

template = PromptTemplate(
    template=" Your are a senior {engineer} so behave like it  now give me the detailed summary on the {paper} give me authentic information about it . Give me the deatiled explaination ", 
    input_variables=['engineer', 'paper'] 

)

chain =  template | model | parser

result = chain.invoke({"engineer":"AI engineer" , "paper": "Attention is all you need"})
print(result)
chain.get_graph().print_ascii()
