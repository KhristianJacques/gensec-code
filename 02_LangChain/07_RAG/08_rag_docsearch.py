from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

import readline
import os

# Open the persisted RAG database with the same embedding function used to build it.
vectorstore = Chroma(
    embedding_function=VertexAIEmbeddings(
        model_name="gemini-embedding-001",
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location="us-west1"
    ),
    persist_directory="./rag_data/.chromadb"
)

def search_db(query):
    """Search the vector database and print the closest matching source."""
    # Run similarity search and show the closest source document.
    docs = vectorstore.similarity_search(query)
    print(f"Query database for: {query}")
    if docs:
        print(f"Closest document match in database: {docs[0].metadata['source']}, count={len(docs)}")
    else:
        print("No matching documents")

print("RAG database initialized.")
retriever = vectorstore.as_retriever()

# List indexed sources before accepting search queries.
document_data_sources = set()
for doc_metadata in retriever.vectorstore.get()['metadatas']:
    document_data_sources.add(doc_metadata['source']) 
for doc in document_data_sources:
    print(f"  {doc}")

print("This program queries documents in the RAG database that are similar to whatever is entered.")
while True:
    line = input(">> ")
    if line:
            # Search the vector database for each entered query.
            search_db(line)
    else:
        break
