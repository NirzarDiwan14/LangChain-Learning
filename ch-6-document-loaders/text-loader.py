from langchain_community.document_loaders import TextLoader
loader = TextLoader("temp.txt",encoding="utf-8")
docs = loader.load()

for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
