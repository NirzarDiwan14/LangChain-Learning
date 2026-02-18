from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader

pdf_loader = DirectoryLoader(
    path = "loader_examples/",
    loader_cls=PyPDFLoader,
    glob = "*.pdf"
    )
txt_loader = DirectoryLoader(
    path = "loader_examples/",
    loader_cls=TextLoader,
    glob = "*.txt"
    )
pdf_docs = pdf_loader.load()
txt_docs = txt_loader.load()

print(len(pdf_docs),len(txt_docs))

for doc in txt_docs:
    print(doc)
    print("----" *20)