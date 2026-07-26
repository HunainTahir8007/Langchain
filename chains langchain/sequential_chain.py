from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite") 

parser = StrOutputParser()

template1 = PromptTemplate(
    template="You act like a senior researcher in the field of {domain}. "
             "Deeply explain the paper '{paper}' with proper intuition. "
             "Stay focused on the domain and avoid unnecessary information.",
    input_variables=["domain", "paper"]
)

template2 = PromptTemplate(
    template="""
Summarize this paper in 10 lines.
Include:
1. Three core points.
2. One final line giving the overall takeaway.

{text}
""",
    input_variables=["text"]
)

chain = template1 | model | parser | template2 | model | parser 

result =  chain.invoke({"domain":"Artificial Intelligence" , "paper" : "Attention is all you need"  })
print(result)