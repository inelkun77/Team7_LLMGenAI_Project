# app/ingest.py

import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

from .config import (
    DOCS_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents():
    docs_dir = Path(DOCS_PATH)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir.resolve()}")

    # Charge PDF + fichiers texte de manière générique
    loaders = []

    # Tous les PDF
    loaders.append(
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
    )

    # Tous les .txt
    loaders.append(
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
    )

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} raw documents.")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore():
    docs = load_documents()
    chunks = split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vectorstore saved to {VECTORSTORE_PATH}")


if __name__ == "__main__":
    build_vectorstore()
