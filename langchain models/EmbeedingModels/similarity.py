from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeed = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')

documents = [
"Virat Kohli  One of the greatest modern batters, known for his consistency and aggressive leadership.",
"Babar Azam  Renowned for his elegant batting technique and exceptional consistency across formats.",
"Ben Stokes  A world-class all-rounder famous for delivering match-winning performances under pressure.",
"Kane Williamson  Widely respected for his calm captaincy and technically sound batting.",
"Jasprit Bumrah An elite fast bowler celebrated for his unique action and deadly accuracy."
]
query = "Tell me about the Ben stokes"

docs  = embeed.embed_documents(documents)
que = embeed.embed_query(query)

print(cosine_similarity([que] , docs))

scores = cosine_similarity([que] , docs)[0]

print(sorted(list(enumerate(scores)), key= lambda x :x[1])[-1])