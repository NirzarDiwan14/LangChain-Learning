from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda

load_dotenv()


model1 = ChatGroq(
    model = "llama-3.3-70b-versatile"
)
class Feedback(BaseModel):
    
    sentiment:Literal["positive","negative","neutral"] = Field(description="The sentiment of the review_text from the given values.")

str_parser = StrOutputParser()
py_parser = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template = "Classify the sentiment of the review: {review_text},{format_intruction}",
    input_variables= ["review_text"],
    partial_variables= {"format_intruction" : py_parser.get_format_instructions()},
)

classifier_chain = prompt1 | model1 | py_parser 

prompt2 = PromptTemplate(
    template = "Write the appropriate feedback to this positive review: {review_text}",
    input_variables= ["review_text"],
)
prompt3 = PromptTemplate(
    template = "Write the appropriate feedback to this negative review: {review_text}",
    input_variables= ["review_text"],
)
positive_chain = prompt2 | model1 | str_parser
negative_chain = prompt3 | model1 | str_parser
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_chain),
    (lambda x: x.sentiment == "negative", negative_chain),
    RunnableLambda(lambda x: "neutral sentiment found")
)

chain = classifier_chain | branch_chain

positive_review_text = """
This is a good screen LED TV that offers great value for money. The HD picture quality is sharp and vibrant, the sound is decent, and the overall build quality is solid.
"""
negative_review_text = """
LED tv i have bought has certain issues in the screen, it offs after some time , sound is on but screen is blank. i dont recommend anyone to purchase this product, totally waste of money  
"""
neutral_review_text = """
This is a good screen LED TV that offers great value for money. The HD picture quality is sharp and vibrant, the sound is decent, and the overall build quality is solid.

However, the major drawback is the processor and overall speed. The TV takes noticeable time to start up, and switching between different OTT apps is sluggish. TCL should definitely work on improving this aspect — ideally by upgrading the RAM to 1.5GB or 2GB for smoother performance.

If you’re looking for a budget-friendly TV with good visuals and sound, it’s a decent choice, but be prepared for some lag during use.
"""

review_texts = [positive_review_text,negative_review_text,neutral_review_text]
for review in review_texts:
    print("----" *20)
    print(review)
    response = chain.invoke({"review_text" : review})
    print(response)
    print("----" *20)