import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "chromadb_store"
collection_name = "documents"

os.makedirs(persist_directory, exist_ok=True)

# Always create client (lightweight)
client = chromadb.Client(Settings(
    persist_directory=persist_directory,
    anonymized_telemetry=False,
    allow_reset=True
))

# Lazy load heavy components
def get_chroma():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory
    )

def store_to_chroma(session_id, text):
    chroma = get_chroma()
    chroma.add_texts([text], metadatas=[{"session_id": session_id}])

def query_chroma(session_id, query):
    chroma = get_chroma()
    docs = chroma.similarity_search(query, k=3)
    filtered_docs = [doc.page_content for doc in docs if doc.metadata.get("session_id") == session_id]
    return "\n\n".join(filtered_docs) if filtered_docs else "No relevant content found."
