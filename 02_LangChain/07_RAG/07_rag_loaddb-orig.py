from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader, DirectoryLoader, TextLoader, PyPDFDirectoryLoader, Docx2txtLoader, UnstructuredMarkdownLoader, WikipediaLoader, ArxivLoader, CSVLoader, GithubFileLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import readline

# Open or create the persistent Chroma vector database.
vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query"),
    persist_directory="./rag_data/.chromadb"
)

def load_docs(docs):
    """Split documents into chunks and store their embeddings in Chroma."""
    # Split loaded documents into chunks before embedding and storing them.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    print(f"# splits: {len(splits)}, split0: {len(splits[0].page_content)} chars")

#    vectorstore.add_documents(documents=splits)
    vectorstore.add_documents(documents=[splits[0]])

def load_urls(urls):
    """Load web pages from URLs and add them to the vector database."""
    # Load web pages and pass the resulting documents into the shared pipeline.
    load_docs(AsyncHtmlLoader(urls).load())

def load_wikipedia(query):
    """Load Wikipedia pages for a query and add them to the vector database."""
    # Load a Wikipedia page for the requested topic.
    load_docs(WikipediaLoader(query=query, load_max_docs=1).load())

def load_arxiv(query):
    """Load an arXiv paper and add it to the vector database."""
    # Load an arXiv paper and normalize its source metadata.
    docs = ArxivLoader(query=query, load_max_docs=1).load()
    docs[0].metadata['source'] = f"arxiv:{query}"
    load_docs(docs)

def load_github(file):
    """Load matching GitHub repository files and add them to the database."""
    # Load matching files from GitHub and add their contents to the vector DB.
    loader = GithubFileLoader(
        repo="wu4f/cs475-src",  # the repo name
        branch="main",  # the branch name
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(file)
        # any file in the repo that ends with file 
    )
    documents = loader.load() 
    load_docs(documents)

def load_youtube(video_id):
    """Load a YouTube transcript and add it to the vector database."""
    # Convert a transcript into a LangChain document with source metadata.
    transcript = YouTubeTranscriptApi().fetch(video_id)
    text = ' '.join([entry.text for entry in transcript])
    docs = [Document(page_content=text, metadata={'source': f'youtube:{video_id}'})]
    load_docs(docs)

def load_txt(directory):
    """Load text files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader).load())

def load_pdf(directory):
    """Load PDF files from a directory into the vector database."""
    load_docs(PyPDFDirectoryLoader(directory).load())

def load_docx(directory):
    """Load DOCX files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.docx", loader_cls=Docx2txtLoader).load())

def load_md(directory):
    """Load Markdown files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader).load())

def load_csv(directory):
    """Load CSV files from a directory into the vector database."""
    load_docs(DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader).load())

# Load selected external and local sources into the vector database.
urls = ["https://www.pdx.edu/academics/programs/undergraduate/computer-science", "https://www.pdx.edu/computer-science/"]
print(f"Loading: {urls}")
load_urls(urls)

wiki_query = "LangChain"
print(f"Loading Wikipedia pages on: {wiki_query}")
load_wikipedia(wiki_query)

arxiv_query = "2310.03714"
print(f"Loading arxiv document: {arxiv_query}")
load_arxiv(arxiv_query)

github_file = "butcher.py"
print(f"Loading github file(s) with ending: {github_file}")
load_github(github_file)

youtube_video_id = "78600iosmis"
print(f"Loading YouTube video: {youtube_video_id}")
load_youtube(youtube_video_id)

text_directory = "rag_data/txt"
print(f"Loading TXT files from: {text_directory}")
load_txt(text_directory)

pdf_directory = "rag_data/pdf"
print(f"Loading PDF files from: {pdf_directory}")
load_pdf(pdf_directory)

docx_directory = "rag_data/docx"
print(f"Loading DOCX files from: {docx_directory}")
load_docx(docx_directory)

md_directory = "rag_data/md"
print(f"Loading MD files from: {md_directory}")
load_md(md_directory)

csv_directory = "rag_data/csv"
print(f"Loading CSV files from: {csv_directory}")
load_csv(csv_directory)

print("RAG database initialized with the following sources.")
retriever = vectorstore.as_retriever()
document_data_sources = set()
for doc_metadata in retriever.vectorstore.get()['metadatas']:
    # Collect unique source names so the user can see what was indexed.
    document_data_sources.add(doc_metadata['source']) 
for doc in document_data_sources:
    print(f"  {doc}")
