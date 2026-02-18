from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = "Generate 3 Facts about the topic: {topic}",
    input_variables= ["topic"],
)

model = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

parser = StrOutputParser()


chain = prompt | model | parser 

response = chain.invoke({"topic" : "deep Learning"})
print(response)