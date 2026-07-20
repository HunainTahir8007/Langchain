from langchain_huggingface import  ChatHuggingFace  ,  HuggingFaceEndpoint
from dotenv import load_dotenv
hf_token = ""


llm = HuggingFaceEndpoint(repo_id = 'meta-llama/Llama-3.1-8B-Instruct' , task ='text-generation' , huggingfacehub_api_token=hf_token)

model = ChatHuggingFace(llm = llm , temperature = 0.7 )

response =model.invoke("Can u suggest me the bast carrier option in the field of IT")

print(response.content)