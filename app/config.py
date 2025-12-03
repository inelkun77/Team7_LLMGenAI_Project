# app/config.py

OLLAMA_MODEL = "llama3"  # ou un autre modèle que tu as pull
EMBEDDING_MODEL = "llama3"  # OllamaEmbeddings utilisera ce nom
VECTORSTORE_PATH = "storage/vectorstores/esilv_faiss"
DOCS_PATH = "data/esilv_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
