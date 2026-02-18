from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda,RunnablePassthrough

load_dotenv()


model1 = ChatGroq(
    model = "llama-3.3-70b-versatile"
)
model2 = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

str_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a short simple summary of formulas from the given topic: {topic}",
    input_variables= ["topic"],
)

prompt2 = PromptTemplate(
    template = "Write the proper vizual explaination for the topic: {topic}",
    input_variables= ["topic"],
)
prompt3 = PromptTemplate(
    template = """
        Merge the following into a proper readable format:

        Topic: {topic}

        Formulas:
        {formulas}

        Visual Explanation:
        {explaination}
    """,
    input_variables= ["topic","formulas","explaination"],
)

runnable_chain = RunnableParallel(
    {   
        "topic": RunnableLambda(lambda x: print("DEBUG:", x["topic"]) or x["topic"]),
        "formulas": prompt1 | model1 | str_parser,
        "explaination" : prompt2 | model2 | str_parser
    }
)
merge_chain = prompt3 | model1 | str_parser

chain = runnable_chain | merge_chain

topic = "Deep Learning"
for chunk in runnable_chain.stream({"topic": "Deep Learning"}):
    print(chunk)

# print("----" *20)
# response = chain.invoke({"topic" : topic})
# print(response)
# print("----" *20)