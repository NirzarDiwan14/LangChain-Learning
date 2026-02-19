# 🦜 LangChain & LangGraph: A Comprehensive Learning Journey

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-1.2.10-blue?style=for-the-badge&logo=chainlink" alt="LangChain">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-orange?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

## 🚀 Overview

Welcome to the ultimate learning repository for **LangChain** and **LangGraph**. This project is designed as a structured curriculum to take you from "Hello World" LLM calls to building production-grade agentic workflows.

> [!TIP]
> This repository uses **uv** for lightning-fast dependency management. Make sure you have it installed!

---

## 🗺️ Learning Roadmap (The Curriculum)

The content is organized into three progressive levels:

### 🟢 Level 1: Foundations
*Master the building blocks of LLM applications.*

| Chapter | Focus | Key Highlights |
| :--- | :--- | :--- |
| **[ch-1-basics](./ch-1-basics)** | Model Integration | [Gemini](./ch-1-basics/gemini_call.py) & [Groq](./ch-1-basics/groq_call.py) setup. |
| **[ch-2-prompts](./ch-2-prompts)** | Prompt Engineering | `ChatPromptTemplate`, Static & Dynamic prompts. |
| **[ch-3-structured_output](./ch-3-structured_output)** | Schema Control | Forcing LLMs to return JSON/Pydantic objects. |
| **[ch-4-outputparsers](./ch-4-outputparsers)** | Parsing Responses | Converting raw text into structured data. |
| **[ch-5-chains](./ch-5-chains)** | Workflow Logic | `RunnableParallel`, `RunnableBranch`, and Sequential Chains. |

### 🟡 Level 2: Data & RAG
*Teach your LLM about private data.*

- **Data Processing**: [Document Loaders](./ch-6-document-loaders) (PDF, Web, TXT) and [Text Splitters](./ch-7-text-splitters).
- **Vector Intelligence**: [Embeddings](./ch-8-embeddings) and [Vector Stores](./ch-9-vector-stores) (ChromaDB, FAISS).
- **Retrieval-Augmented Generation**: 
    - [Basic RAG](./ch-11-rags/basic_rag.py)
    - [Memory-Aware RAG](./ch-11-rags/memory_rag.py)
    - [Database-Backed RAG](./ch-11-rags/db_rag.py)

### 🔴 Level 3: Agents & Graphs
*Building autonomous, stateful AI systems.*

- **[ch-12-Agents](./ch-12-Agents)**: Creating agents with [custom tools](./ch-12-Agents/tools.py).
- **[ch-13-langgraph](./ch-13-langgraph)**: The evolution of agents using LangGraph:
    - **Basics**: Nodes, Edges, and State.
    - **Parallelization**: Running multiple tasks in parallel for speed.
    - **Architecture**: Orchestrator-Worker patterns for complex problem solving.

---

## 🛠️ Setup Guide

### 1. Prerequisites
- **Python 3.13+**
- **uv** (Install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 2. Installation
```bash
git clone https://github.com/your-username/LangChain-Learning.git
cd LangChain-Learning
uv sync
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY="your_api_key"
GROQ_API_KEY="your_api_key"
HUGGINGFACEHUB_API_TOKEN="your_token"
```

> [!IMPORTANT]
> Never commit your `.env` file! It's already included in `.gitignore`.

---

## 📂 Featured Showcase

-   **[Projects/blog_post_generator.py](./Projects/blog_post_generator.py)**: A full pipeline that takes a topic and produces a formatted blog post.
-   **[basic_rag_chroma_db](./basic_rag_chroma_db)**: A specialized implementation for enterprise-grade PDF querying.

---

## 🧪 Tech Stack

| Type | Technology |
| :--- | :--- |
| **Frameworks** | LangChain, LangGraph, Streamlit |
| **LLMs** | Google Gemini, Llama 3 (via Groq), HuggingFace |
| **Vector DB** | ChromaDB, FAISS |
| **Tools** | Wikipedia API, DuckDuckGo Search, Arxiv |

---

## 🤝 Contributing

Found a bug or have a new example? Feel free to open a PR! Let's build the best LangChain learning resource together.

---
*Made with ❤️ for the AI Community*
