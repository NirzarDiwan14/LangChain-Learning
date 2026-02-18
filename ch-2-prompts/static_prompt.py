from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


static_prompt = PromptTemplate(
    input_variables = [],
    template = "Write a universal truth."
)

prompt_text = static_prompt.format()
print(prompt_text)



llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)

response = llm.invoke(prompt_text)
print(response.content)