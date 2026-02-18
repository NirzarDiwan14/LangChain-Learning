from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("CrewAI MasterClass.pdf")

pdf_docs = loader.load()

for doc in pdf_docs:
    print(doc.page_content)
    print("------" * 20)