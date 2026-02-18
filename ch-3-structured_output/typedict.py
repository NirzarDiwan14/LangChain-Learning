from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


class Review(TypedDict):
    summary: str
    sentiment: str


review_text = """
This is a good screen LED TV that offers great value for money. The HD picture quality is sharp and vibrant, the sound is decent, and the overall build quality is solid.

However, the major drawback is the processor and overall speed. The TV takes noticeable time to start up, and switching between different OTT apps is sluggish. TCL should definitely work on improving this aspect — ideally by upgrading the RAM to 1.5GB or 2GB for smoother performance.

If you’re looking for a budget-friendly TV with good visuals and sound, it’s a decent choice, but be prepared for some lag during use.
"""

model = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

structured_model = model.with_structured_output(Review)

response = structured_model.invoke(review_text)
for key,value in response.items():
    print(key,value)