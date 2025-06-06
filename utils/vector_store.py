import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "chromadb_store"
os.makedirs(persist_directory, exist_ok=True)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

client = chromadb.Client(Settings(persist_directory=persist_directory))

collection_name = "documents"

# Initialize or get the collection
chroma = Chroma(
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=persist_directory
)

def store_to_chroma(session_id, text):
    chroma.add_texts([text], metadatas=[{"session_id": session_id}])

def query_chroma(session_id, query):
    docs = chroma.similarity_search(query, k=3)
    filtered_docs = [doc.page_content for doc in docs if doc.metadata.get("session_id") == session_id]
    return "\n\n".join(filtered_docs) if filtered_docs else "No relevant content found."
