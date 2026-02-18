from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ Create prompt template
prompt = PromptTemplate(
    template="give the answer of the question: {question} from the given text: {text}.",
    input_variables=["question","text"],
)

# 2️⃣ Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# 3️⃣ Load web page
url = "https://docs.langchain.com/"
url = "https://qdrant.tech/documentation/overview/"
loader = WebBaseLoader(url)
docs = loader.load()

# 4️⃣ Format prompt using PromptTemplate
final_prompt = prompt.format(
    question="What is this content about?",
    text=docs[0].page_content
)

# 5️⃣ Call LLM
response = llm.invoke(final_prompt)

# 6️⃣ Print output
print(response)
