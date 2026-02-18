from dotenv import load_dotenv 
load_dotenv()


#1 load a document
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("loader_examples/sample.pdf")
documents = loader.load()


#2 splitting the documents 
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 40,
)

docs = splitter.split_documents(documents)

#3 create embeddings 

from langchain_huggingface import HuggingFaceEndpointEmbeddings
embedding_model = HuggingFaceEndpointEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2",
)

# 4 store embeddings into vector db
from langchain_chroma import Chroma
vector_store = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model,
    collection_name="basic_rag",
    persist_directory="./basic_rag_chroma_db",
    )

#5 create a retriever 

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 2,"lambda_mult": 0.5}
)

#6  Initialize LLM

from langchain_groq import ChatGroq


model = ChatGroq(
     model="llama-3.1-8b-instant",
     temperature=0
)

#7 create prompt template 
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template = """
        You are an AI assistant,Use the following context to answer the question.
        if the answer is not present in the context, say you don't know.
        context: {context}
        question: {question}
    """,
    input_variables=["context","question"]
    )

#8 Output parser definition 
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()


# 9 build a rag chain 


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    } | prompt | model | parser
)


query = "What is the total population of india right now?"
response = rag_chain.invoke(query)

print(response)