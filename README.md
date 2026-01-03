# ESILV Chatbot – RAG Architecture

## Project Overview
This project implements an intelligent chatbot designed for **ESILV** and the broader **Pôle Léonard de Vinci** ecosystem (ESILV, EMLV, IIM). The chatbot provides reliable, contextual, and domain-specific answers to questions related to admissions, academic programs, student life, and administrative procedures.

The system is based on a **Retrieval-Augmented Generation (RAG)** architecture combined with **specialized agents**, ensuring factual accuracy while minimizing hallucinations commonly associated with standalone Large Language Models (LLMs).

This project was developed as part of the **LLM & GenAI – A5 DIA4** course.

---

## Key Features
- Retrieval-Augmented Generation (RAG) using FAISS
- Specialized agents by domain (Admissions, Academics, Student Life, Administration)
- Deterministic routing via lexical keyword analysis
- Low-temperature LLM generation for factual responses
- Support for PDFs, text files, web content, and user-uploaded documents
- Feedback collection via Streamlit interface

---

## System Architecture

### Global Pipeline
```
User Query
   ↓
AgentRouter (router.py)
   ↓
Specialized Agent
   ↓
RAG Chain (rag.py)
   ↓
FAISS VectorStore + LLM (Ollama)
   ↓
Generated Response
```

### Specialized Agents
- **AdmissionsAgent** – Application processes, requirements, deadlines
- **AcademicsAgent** – Programs, courses, academic projects
- **StudentLifeAgent** – Associations, campus life, events
- **AdminAgent** – Certificates, absences, internal procedures (fallback agent)

---

## Project Structure
```
project-root/
│
├── config.py #Central configuration and hyperparameters
├── ingest.py #Data ingestion, cleaning, chunking, embeddings
├── rag.py #RAG chain implementation
├── agents.py #Domain-specific agents and prompts
├── router.py #Keyword-based question routing
├── data/ #PDFs, text files, scraped content
├── vectorstore/ #FAISS index storage
├── app.py #Streamlit application entry point
├── votes.csv #User feedback logging
└── README.md
```

---

## ⚙️ Technical Configuration

| Parameter | Value |
|---------|-------|
| Chunk size | 400 characters |
| Chunk overlap | 60 characters |
| Retriever k | 4 documents |
| Embedding model | nomic-embed-text |
| LLM | gemma3:1b (via Ollama) |
| Temperature | 0.2 |
| Max context | 2000 characters |
| Max response length | 8 lines |

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Ollama installed and running
- Git

### Install dependencies
```bash
pip install -r requirements.txt
```

### Pull required models
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

---

## Data Ingestion

To ingest and index documents:
```bash
python ingest.py
```

Supported sources:
- PDF files
- Text files
- Web pages (via custom scrapers)

All documents are cleaned, chunked, embedded, and indexed into FAISS.

---

## Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```

Users can:
- Ask questions in natural language
- Upload PDF or text files for additional context
- Rate responses (👍 / 👎)

---

## Evaluation & Feedback

- Manual evaluation focused on relevance, factual grounding, and domain compliance
- User feedback is collected and stored in `votes.csv`
- Feedback can be used to improve routing, prompts, and document coverage

---

## Current Limitations
- Keyword-based routing sensitive to phrasing and synonyms
- No automated confidence or retrieval quality metrics
- Latency increases with large uploaded documents
- Limited to text and PDF file uploads

---

## Future Improvements
- Embedding-based or probabilistic routing
- Adaptive chunking strategies
- Document reranking with cross-encoders
- Source citation in responses
- Multilingual support
- Scalability optimizations for large corpora

---

## Team
**Team 7 – ESILV**  
LLM & Generative AI – A5 DIA4

**Members:**
- LAGZOULI Lina
- LADRAA Lamia
- MOUTON Cyprien
- MAHCER Neil

