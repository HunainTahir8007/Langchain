
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model = "gemini-2.5-flash")

response = llm.invoke("Tell me three line about the pakistan??")
print(response) 