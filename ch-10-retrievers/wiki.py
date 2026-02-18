from langchain_community.retrievers import WikipediaRetriever,WebResearchRetriever
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSearchAPIWrapper

wiki_retriever = WikipediaRetriever(
    top_k_results=3,
    lang="en"
)

# web_retriever = WebResearchRetriever.from_llm(
#     vectorstore=Chroma,
#     llm=ChatGroq( model="llama-3.3-70b-versatile",),
#     num_search_results=3,
#     search = GoogleSearchAPIWrapper()
# )


query = "Artificial Intelligence"

results = wiki_retriever.invoke(query)
# results = web_retriever.invoke(query)
for res in results:
    print(res)
    print("-----"  *20)