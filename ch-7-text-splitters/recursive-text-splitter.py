from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter,PythonCodeTextSplitter

text = """
Chapter 1: Introduction

LangChain is an open-source framework designed to help developers build applications powered by large language models (LLMs). It provides tools for connecting LLMs with data, APIs, and memory. With LangChain, you can create chatbots, agents, and custom AI applications that can reason over structured and unstructured data.

Section 1.1: Background

In recent years, large language models have gained significant attention due to their ability to generate human-like text. However, using LLMs in applications requires managing context, integrating external data sources, and ensuring outputs are accurate and reliable.

Chapter 2: Core Concepts

LangChain introduces several core concepts:

- Chains: Sequences of calls or prompts for reasoning and processing.
- Agents: Systems that can take actions in the real world based on LLM outputs.
- Memory: Mechanisms to store and retrieve information across multiple interactions.
- Prompts: Templates and instructions that guide LLM behavior.

Section 2.1: Example

For instance, a chatbot can use a chain to process a user query, access a database for relevant facts, and then generate a contextually appropriate answer using a prompt template.

Chapter 3: Conclusion

LangChain simplifies the development of intelligent applications by abstracting complex workflows into reusable components. Its modular design allows developers to focus on building logic rather than handling low-level LLM details.

"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
)

results = splitter.split_text(text)

print(len(results))
for chunk in results:
    print(chunk)
    # print("metadata: ",chunk.metadata)
    # print("page_content: ",chunk.page_content)
    print("-----"* 30)