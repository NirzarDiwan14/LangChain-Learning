from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)

response = llm.invoke("What is the capital of sri lanka?")
print(response.content)