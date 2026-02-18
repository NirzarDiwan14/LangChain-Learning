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

parser = JsonOutputParser()

template_1 = PromptTemplate(
    template = "give me the name age and city of the fictional character, it has to be indian. {format_instruction}",
    input_variables= [],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template_1 | model | parser 

response = chain.invoke({"topic" : "black hole"})
print(response)