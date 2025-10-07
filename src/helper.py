from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents
 
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if doc.page_content.strip():  # only keep docs with non-empty content
            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": src}
                )
            )
    return minimal_docs  # <- RETURN AFTER LOOP
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk
 
def download_hugging_face_embeddings():
  
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
       # model_kwargs = {"device": "cuda" if torch.cuda.is_available()else "cpu"}
    )
    return embeddings

