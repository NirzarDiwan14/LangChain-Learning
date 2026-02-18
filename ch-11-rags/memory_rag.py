from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from typing import List
from langchain_core.messages import AIMessage,HumanMessage,BaseMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()


loader = PyPDFLoader("loader_examples/sample.pdf")
documents = loader.load()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
)
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

chunks = splitter.split_documents(documents)

embedding = HuggingFaceEndpointEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2",
)


vector_store = Chroma.from_documents(
    collection_name="memory_rag",
    persist_directory="./basic_rag_chroma_db",
    documents=chunks,
    embedding=embedding
)

retriever = vector_store.as_retriever(
    search_kwargs = {"k": 2}
)



# prompt = ChatPromptTemplate(
#     SystemMessage(
#         content="""
# You are a helpful ai assistant.Answer Strictly from the provided context.
# if the answer is not present in the context then just say "I don't Know". 
#         """
#     ),
#     MessagesPlaceholder(variable_name = "chat_history"),
#     (
#         "humam",
#         "Context: \n {context} \n\n question: \n {input}"
#     )
# )


prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content="""
You are a helpful AI assistant. Answer strictly from the provided context.
If the answer is not present in the context, then just say "I don't know".
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context: \n {context} \n\n question: \n {input}")
    ]
)




def conversational_rag(user_input: str,chat_history: List[BaseMessage]):
    docs = retriever.invoke(user_input)
    context = "\n\n".join(
        f"[Page {d.metadata.get("page","N/A")}] \n {d.page_content}" for d in docs
    )
    messsages = prompt.invoke(
        {
            "input": user_input,
            "context" : context,
            "chat_history" : chat_history 
        }
    )

    response = llm.invoke(messsages)

    return response,docs 


chat_history: List[BaseMessage] = []

print("<<<<<<<CONVERSATIONAL RAG SYSTEM>>>>>>>>>>")


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content = user_input))
    responses, sources = conversational_rag(user_input,chat_history)
    
    chat_history.append(AIMessage(content = responses.content))

    print("\nAI:",responses.content)