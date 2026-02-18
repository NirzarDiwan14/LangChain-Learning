from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from dotenv import load_dotenv
load_dotenv()



chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant which provides info about {subject}."),
    HumanMessagePromptTemplate.from_template("Can you tell me something amazing about {subject}?")
])


prompt = chat_prompt.format_messages(subject = "OCR in ADE")

print(prompt)










llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0,

)

response = llm.invoke(prompt)
print(response.content)