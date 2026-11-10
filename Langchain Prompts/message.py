from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.5)

messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Tell me 5 line about pakistan")
    
]
result = model.invoke(messages)
messages.append(AIMessage(result.content))
print(result.content)
print(messages)