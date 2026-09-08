import os
import readline
from langchain_classic import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import chainlit as cl

# Configure the chat model used by the Chainlit RAG app.
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

@cl.on_chat_start
async def on_chat_start():
    """Send the initial Chainlit logo and welcome message."""
    # Send branding and a short welcome when the Chainlit session opens.
    logo = cl.Image(name="logo", display="inline", url="https://codelabs.cs.pdx.edu/images/pdx-cs-logo.png")
    await cl.Message(content="", elements=[logo]).send()

    welcome_text = (
        "**Welcome to the PSU Generative Security Chatbot!**\n\n"
        "Ask me anything from my set of documents.\n\n"
    )
    await cl.Message(content=welcome_text).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle a Chainlit user message by returning a RAG answer."""
    # Answer each chat message through the RAG chain.
    user_query = message.content
    answer = rag_chain.invoke(user_query)
    await cl.Message(content=answer).send()

if __name__ == "__main__":
    cl.run()
