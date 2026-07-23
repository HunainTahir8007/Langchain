from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st 

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-2.5-flash" )


st.header("Research Paper Summerization TOOL")
user_input =st.text_input("Enter the prompt")

if st.button("Summerize"):
    result = model.invoke(user_input)
    st.write(result)