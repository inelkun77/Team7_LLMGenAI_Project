# app/ingest.py

import os
import json
import re
from pathlib import Path

from langchain_core.documents import Document
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
    WEB_JSONL_PATH,
)




def clean_web_text(raw: str) -> str:
    if not raw:
        return ""

    raw = raw.replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)

    lines = [ln.strip() for ln in raw.split("\n")]
    cleaned = []

    drop_if_contains = [
        "gestion de la vie privée",
        "manage consent",
        "save my preferences",
        "tout refuser",
        "ok, j'accepte",
        "cookies",
        "cookie",
        "mentions légales",
        "plan d’accès",
    ]

    short_noise = {
        "français", "english", "paris", "nantes", "montpellier",
        "madame", "monsieur", "bac+1", "bac+2", "bac+3", "bac+4", "bac+5",
    }

    for ln in lines:
        if not ln:
            continue
        lnl = ln.lower()
        if lnl in short_noise:
            continue
        if any(x in lnl for x in drop_if_contains):
            continue
        if len(ln) < 25:
            continue
        cleaned.append(ln)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text



def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)   
    return text.strip()


def is_useful_url(url: str) -> bool:
    if not url:
        return False

    u = url.lower()

    bad = [
        "mentions-legales", "politique", "cookie", "cookies",
        "agenda", "presse", "recrutement", "plan-dacces",
    ]
    if any(b in u for b in bad):
        return False

    return any(
        k in u for k in [
            "formations", "admissions", "cycle-ingenieur",
            "bachelor", "msc", "majeure", "vie-etudiante", "campus"
        ]
    )


def detect_school(url: str) -> str:
    if "esilv.fr" in url:
        return "esilv"
    if "emlv.fr" in url:
        return "emlv"
    if "iim.fr" in url:
        return "iim"
    return "unknown"


def detect_type(url: str) -> str:
    u = url.lower()
    if any(k in u for k in ["admission", "candidature", "postuler"]):
        return "admissions"
    if any(k in u for k in ["vie-etudiante", "association", "campus", "bde"]):
        return "student_life"
    return "academics"


def load_documents():
    docs_dir = Path(DOCS_PATH)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir.resolve()}")

    documents = []


    pdf_loader = DirectoryLoader(str(docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()


    for doc in pdf_docs:
        doc.page_content = clean_pdf_text(doc.page_content)[:3000]

    documents.extend(pdf_docs)


    txt_loader = DirectoryLoader(str(docs_dir), glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())


    jsonl_path = Path(WEB_JSONL_PATH)

    if jsonl_path.exists():
        added, skipped = 0, 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                except Exception:
                    skipped += 1
                    continue

                url = obj.get("url", "")
                raw_text = (obj.get("text") or "").strip()
                title = obj.get("title", "")

                if not raw_text or not is_useful_url(url):
                    skipped += 1
                    continue

                text = clean_web_text(raw_text)
                if len(text) < 250:
                    skipped += 1
                    continue

                school = detect_school(url)
                doc_type = detect_type(url)

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "web",
                            "url": url,
                            "title": title,
                            "school": school,
                            "type": doc_type,
                        },
                    )
                )
                added += 1

        print(f"Loaded {added} web pages ({skipped} skipped).")

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
