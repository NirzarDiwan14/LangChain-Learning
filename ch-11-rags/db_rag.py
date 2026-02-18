import streamlit as st
import sqlite3
import uuid
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage, 
    AIMessage,
    SystemMessage
)
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

load_dotenv()

# database

conn=sqlite3.connect("chat_memory1.db", check_same_thread=False)
cursor=conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_history(
    session_id TEXT,
    role TEXT,
    content TEXT
    )
    """
)

conn.commit()