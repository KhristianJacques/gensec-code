"""A small Wikipedia-backed document loader."""

from __future__ import annotations

import wikipedia
from datetime import datetime, timedelta
from langchain_core.documents import Document


class WikipediaLoader:
    """Load Wikipedia pages matching a search query.

    Network access is deliberately deferred until :meth:`load` so constructing a
    loader is inexpensive and side-effect free.
    """

    def __init__(self, query: str, load_max_docs: int = 4) -> None:
        """Store the search query and validate the requested document count."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if load_max_docs < 1:
            raise ValueError("load_max_docs must be at least 1")

        self.query = query
        self.load_max_docs = load_max_docs

    def load(self) -> list[Document]:
        """Search Wikipedia and return up to ``load_max_docs`` page documents."""
        # wikipedia==1.4.0 defaults to an HTTP API URL.  Wikimedia's API requires
        # HTTPS, so update the package setting before its first request.
        if wikipedia.wikipedia.API_URL.startswith("http://"):
            wikipedia.wikipedia.API_URL = wikipedia.wikipedia.API_URL.replace(
                "http://", "https://", 1
            )
            wikipedia.API_URL = wikipedia.wikipedia.API_URL
        wikipedia.wikipedia.USER_AGENT = (
            "WikipediaLoader/1.0 (LangChain educational example;contact@fau.edu)"
        )
        wikipedia.USER_AGENT = wikipedia.wikipedia.USER_AGENT

        wikipedia.set_rate_limiting(True, min_wait=timedelta(milliseconds=500))

        # Search for matching page titles, then fetch each full page.
        titles = wikipedia.search(self.query, results=self.load_max_docs)
        documents: list[Document] = []

        for title in titles:
            page = wikipedia.page(title, auto_suggest=False)
            documents.append(
                Document(
                    page_content=page.content,
                    metadata={
                        "source": page.url,
                        "title": page.title,
                        "summary": page.summary,
                    },
                )
            )

        return documents

if __name__ == "__main__":
    # Live example: requires internet access.
    loader = WikipediaLoader(query="LangChain", load_max_docs=4)
    documents = loader.load()

    # Print metadata so the loaded sources can be inspected quickly.
    for index, document in enumerate(documents, start=1):
        print(f"Document {index} metadata:")
        print(document.metadata)
