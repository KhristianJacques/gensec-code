import os
import importlib
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure the model used by the imported page-query helper.
llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"),temperature=0)
#from langchain_openai import ChatOpenAI
#llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))
#from langchain_anthropic import ChatAnthropic
#llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL"))

# Reuse query_page from the loader/transformer example.
module = importlib.import_module("01_loaders_transformers")

# Load a large page and split it into overlapping chunks.
loader = AsyncHtmlLoader("https://www.pdx.edu/academics/programs/undergraduate/computer-science")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap=1000)
docs_splits = text_splitter.split_documents(docs)
print(f"Split {len(docs[0].page_content)} byte document into {len(docs_splits)}")

# Keep only chunks that contain the phrase needed for the question.
article_chunks = [i for i in range(len(docs_splits)) if 'job placement' in docs_splits[i].page_content]

for i in article_chunks:
    # Send each relevant chunk to the LLM-backed query helper.
    print(f"Found chunk {i} with 'job placement' in it.  Sending to LLM")
    module.query_page(docs_splits[i].page_content)
