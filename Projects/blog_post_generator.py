from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()


print("--" * 20 + "Blog Post Generator" + "--" * 20)
print("Provide ideas or topics for the blog post. OR type 'exit' to Finish")
topic = input("Enter the topic for blog post generator: ")

chat_prompt_template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("you are a professional blog writer. help generate infomative,engaging and well structured blog post about {topic}."),
    HumanMessagePromptTemplate.from_template("Write a detailed blog post about {topic}")
])
chat_history = []


chat_model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0
)
while True:
    user_input = input("<exit for stopping> OR give the ideas ")

    if user_input.lower() == "exit":
        print("Exiting from the blog post generator...")
        break

    messages = chat_prompt_template.format_messages(topic=topic)

    # Add previous history
    messages.extend(chat_history)

    # Add new user message
    user_message = HumanMessage(content=user_input)
    messages.append(user_message)

    response = chat_model.invoke(messages)

    print("Blog post content:\n", response.content)

    # Save to history
    chat_history.append(user_message)
    chat_history.append(AIMessage(content=response.content))
