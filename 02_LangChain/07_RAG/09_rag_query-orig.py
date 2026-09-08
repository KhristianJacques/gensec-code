import os
import readline
from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Configure the chat model used to answer questions from retrieved context.
llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"))

# Open the persisted vector database and create a retriever from it.
vectorstore = Chroma(
     persist_directory="./rag_data/.chromadb",
     embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query")
)

retriever = vectorstore.as_retriever()

# Pull a standard RAG prompt from LangChain Hub.
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    """Join retrieved document contents into one prompt context string."""
    # Combine retrieved documents into the context string expected by the prompt.
    return "\n\n".join(doc.page_content for doc in docs)

# Retrieve context, fill the prompt, call the model, and parse plain text.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Welcome to my RAG application.  Ask me a question and I will answer it from the documents in my database shown below")
# Iterate over documents and dump metadata
document_data_sources = set()
for doc_metadata in retriever.vectorstore.get()['metadatas']:
    # Collect unique source names so the user can see what can be queried.
    document_data_sources.add(doc_metadata['source']) 
for doc in document_data_sources:
    print(f"  {doc}")

while True:
    line = input("llm>> ")
    if line:
        # Answer each question using the retrieval-augmented chain.
        result = rag_chain.invoke(line)
        print(result)
    else:
        break
