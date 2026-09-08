# Generative Security Class Examples

This repository contains Python examples for a the Security Engineering with Generative AI course at FAU adapted from the git@github.com:wu4f/cs475-src.git repository by Professor Wu-chang Feng from pdx.edu. 
The code is organized as a sequence of class modules that introduce model providers, LangChain primitives, retrieval-augmented generation, and agent/tool workflows.

This repository is being updated by the course instructor to be compatible with current software versions. 
New directories will be made available in time for lab and homework assignments. 

## Repository Layout as of 8/26/26

```text
.
├── 01_Intro/                 # Basic model invocation examples
├── 02_LangChain/             # LangChain prompts, messages, chains, parsers, memory
│   └── 07_RAG/               # Document loading, chunking, embeddings, vector search, RAG
└── 03_Agents/                # LangChain tools, SQL toolkits, custom tools, LangGraph agents
```

## Modules

### `01_Intro`

Introductory scripts for calling several chat model providers through LangChain integrations.

- `01_models.py` calls Google Gemini, OpenAI, Anthropic, and xAI models.
- `02_ollama.py` demonstrates local model use with Ollama.
- `03_multi.py` compares responses across multiple providers.

### `02_LangChain`

Core LangChain examples.

- `01_template.py` shows prompt templates.
- `02_messages.py` works with message types.
- `03_context_safety.py` demonstrates safety-related model settings.
- `04_lcel_chains.py` introduces LangChain Expression Language chains.
- `05_memory.py` explores conversational memory.
- `06_parsers_pydantic.py` uses structured output parsing with Pydantic.

### `02_LangChain/07_RAG`

Retrieval-augmented generation examples with multiple document formats and vector search.

- Document loading and transformation examples: `01_loaders_transformers.py`
- Text splitting and chunking examples: `02_chunkers.py`, `03_recursive_loader.py`
- Tokenization and token-cost examples: `04_tokenizers.py`, `05_token_costs.py`
- Embeddings and vector database loading: `06_embeddings.py`, `07_rag_loaddb.py`
- RAG document search and query scripts: `08_rag_docsearch.py`, `09_rag_query.py`
- Chainlit chat interface: `10_chainlit_rag_query.py`
- Sample RAG data lives under `rag_data/` and includes text, Markdown, CSV, PDF, and DOCX files.

### `03_Agents`

Agent and tool examples using LangChain and LangGraph.

- `01_tools_python.py` and `02_tools_builtin.py` demonstrate built-in tool use.
- `03_toolkits_sql.py` uses SQL toolkits with included SQLite databases.
- `04_tools_custom_decorator.py` and `05_tools_custom_pydantic.py` define custom tools.
- `06_langgraph_linux.py`, `07_langgraph_feedback.py`, and `08_langgraph_webassist.py` demonstrate LangGraph workflows.
- `db_data/` contains SQLite databases used by the SQL examples.

## Requirements

Follow the instructions from the Codelabs posted on the [updated Codelabs site](https://icardei.github.io/gensec-web/).
