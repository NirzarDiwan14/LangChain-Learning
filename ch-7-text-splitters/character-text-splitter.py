from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader("loader_examples/sample.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50,
    separator="-"
)

results = splitter.split_documents(docs)

print(len(results))
for chunk in results:
    print("metadata: ",chunk.metadata)
    print("page_content: ",chunk.page_content)
    print("-----"* 30)