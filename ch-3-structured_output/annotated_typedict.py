from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

load_dotenv()


class Review(TypedDict):
    specs: Annotated[list[str],"must write down all the key specifications discussed in a review in a list"]
    summary: Annotated[str,"must write down a brief summary of review."]
    sentiment: Annotated[str,"must return a sentiment either in positive or in negative no in between"]
    pros: Annotated[Optional[list[str]],"write down all the pros inside the list if any."]
    cons: Annotated[Optional[list[str]],"write down all the cons inside the list if any."]


review_text = """
This is a good screen LED TV that offers great value for money. The HD picture quality is sharp and vibrant, the sound is decent, and the overall build quality is solid.

However, the major drawback is the processor and overall speed. The TV takes noticeable time to start up, and switching between different OTT apps is sluggish. TCL should definitely work on improving this aspect — ideally by upgrading the RAM to 1.5GB or 2GB for smoother performance.

If you’re looking for a budget-friendly TV with good visuals and sound, it’s a decent choice, but be prepared for some lag during use.
"""


review_text_2 = """
Marvel's Iron Fist' isn’t just the wimpiest punch ever thrown by the world’s mightiest superhero factory. The new Netflix binge swings and misses so bad that it spins itself around and slaps itself silly with a weirdly flaccid hand.
"""

model = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

structured_model = model.with_structured_output(Review)

response = structured_model.invoke(review_text_2)
for key,value in response.items():
    print(key,value)