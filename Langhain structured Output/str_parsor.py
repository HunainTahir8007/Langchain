from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
hf_token= ""

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct" , task='text-generation' , huggingfacehub_api_token=hf_token )

model = ChatHuggingFace(llm=llm)

tempelate1 =PromptTemplate(
    template= "Write a deatiled note on {topic} " , 
    input_variables=['topic']

)

tempelate2 =PromptTemplate(
    template= "Write a executive 5 line summary  on {text} " , 
    input_variables=['text']

)

prompt1 =tempelate1.invoke({"topic": "black hole"})
result = model.invoke(prompt1)

prompt2 = tempelate2.invoke({"text": result.content})
response = model.invoke(prompt2)

print(response.content)

 