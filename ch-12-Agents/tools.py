from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
llm = ChatGroq(
    model = "llama-3.3-70b-versatile"

)

@tool
def multiply(a,b):
    """Multiply 2 numbers """
    return a * b

@tool
def sqrt(a):
    """give square root of the number """
    import math 
    return math.sqrt(a)

tools = [multiply,sqrt]

system_prompt = """
You are a helpfu ai agent, use tools if required or answer directly.
"""
agent = create_agent(
    tools = tools ,
    model = llm,
    system_prompt=system_prompt,
)


print("<<<<<<<<<<<<Langchain agent>>>>>>>>")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages": [{"role":"user",'content': user_input}]
        }
    )

    print("\nAI:",response['messages'][-1].content)
    print("\n ======agent debug mode ==========",response)
    print("\n\n")