import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the chunks
with open('na_edu_chunks.json', 'r') as f:
    documents = json.load(f)

# Initialize the embeddings model (free alternative to OpenAI)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create texts and metadata
texts = [doc["content"] for doc in documents]
metadatas = [{"source": doc["source"]} for doc in documents]

# Create the vector store
vectorstore = Chroma.from_texts(
    texts=texts, 
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./chroma_db"
)

# Persist the vector store
vectorstore.persist()

print("Vector database created and persisted to ./chroma_db")