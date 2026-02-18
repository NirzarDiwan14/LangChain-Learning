from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
docs = loader.load()

print(len(docs))
print(docs[1].metadata)
print(docs[1].page_content)
