from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


dynamic_prompt = PromptTemplate(
    template = "Write a fact about {topic} in {style} style.",
    input_variables = ["topic","style"],
)

prompt_text = dynamic_prompt.format(topic = "deep learning data",
                                    style = "funny" )
print(prompt_text)



llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)

response = llm.invoke(prompt_text)
print(response.content)