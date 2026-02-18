from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser

load_dotenv()
model = ChatGroq(
    model = "llama-3.3-70b-versatile"
)


template_1 = PromptTemplate(
    template = "Write a detailed report on the {topic}.",
    input_variables= ["topic"]
)
template_2 = PromptTemplate(
    template = "Write a 5 line summary of the text : {text}.",
    input_variables= ["text"]
)
parser = StrOutputParser()
chain = template_1 | model | parser | template_2 | model | parser 

response = chain.invoke({"topic" : "black hole"})
print(response)