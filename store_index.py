from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
import pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY "] = PINECONE_API_KEY 
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_files(r"C:\Users\devko\OneDrive\Desktop\machine-learning-projects\Medical-chatbot\medicalchatbotwith-LLMs-Langchain-pinecone-flask-aws\data")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = "pcsk_6ieGdc_SGmU652RwnVW3zmZ6UzqZQks5C5c4N1iqUj1VadWCWRTde8sukPe2JaPrs8dvSH"
pc = pinecone.Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name= index_name,
        dimension=384,
        metric = "cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
            
        
    )
    index = pc.Index(index_name)
    # The client class is capitalized and is now under the main 'pinecone' module


docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name=index_name
    
)

