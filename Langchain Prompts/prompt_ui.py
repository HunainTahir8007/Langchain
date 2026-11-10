from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash")
paper_input = st.selectbox("select Resarch Paper name " , ["Attention Is all yuo need " , "BERT : Pre Training of Deep Bidirectional Transformers "  , "GPT-3 : Language models are Few - short learners " , "Diffusion Models Beat GANs on Image Synthesis "])
style_input = st.selectbox("Select the Explaination Style " , ["Beginner-Friendly " , "technical "  , "Code-Oriented" , "Mathematical"])
length_input = st.selectbox("Select Explaination Length " , ["Short (1-2 Paragraphs ) " , "Medium (3-5 paragrapghs ) " , "Long (detailed Explaination)"])

template = PromptTemplate(
    template= """Please Summerize the research paper title "{paper_input} wit hte following speacifications 
    Explaination Style :{style_input} 
    Expalination length : {length_input} 
    1. Mathematical Details : 
    -Include relevent mathematical equations if present in the paper 
    - Explain the mathematical concepts using simplle , intutuive coe snippet where applicable .
    2 Analogies : 
    -Use relatable analogies to simplify complex ides 
    
    Ensure theat the summary is clear , accuracte adn the aligned with the provided style length
     """,
     input_variables=["paper_input","style_input" , "length_input"]
)
Prompt =template.invoke({
    "paper_input":paper_input ,
    "style_input":style_input , 
    "length_input" : length_input
})


st.header("Research Paper Summerization Tool") 

if st.button("Summarize"):
    result = model.invoke(Prompt)
    st.write(result)
    