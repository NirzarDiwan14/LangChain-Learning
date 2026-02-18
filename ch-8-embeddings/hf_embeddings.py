from langchain_huggingface import HuggingFaceEndpointEmbeddings

from dotenv import load_dotenv
load_dotenv()


embeddings = HuggingFaceEndpointEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)
query = "Nirzar is a aiml intern"
result = embeddings.embed_query(query)
print(len(result))