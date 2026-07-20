import sys

print(sys.executable)
from langchain_huggingface import HuggingFaceEmbeddings



embeeding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text = "Certainly! Choosing the best career path in IT depends on your interests, skills, and long-term goals. Here are some of the top IT careers in 2024, based on demand, salary, and growth potential"

vector = embeeding.embed_query(text)
print(str(vector))