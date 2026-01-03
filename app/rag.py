# app/rag.py

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from .config import OLLAMA_MODEL, EMBEDDING_MODEL, VECTORSTORE_PATH



# Load FAISS vectorstore

def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )



# Build RAG chain

def build_rag_chain(system_prompt: str, agent_type: str = None):
    """
    RAG final :
    - FAISS reçoit TOUJOURS une string (la question)
    - user_context = document uploadé (temporaire)
    - PAS de dict mal formé
    """

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    vectorstore = load_vectorstore()


    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}#meilleur extraits dans la base FAISS
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question : {question}\n\n"
                "Contexte institutionnel (RAG) :\n{context}\n\n"
                "Document utilisateur (si fourni) :\n{user_context}\n\n"
                "Réponse (français, claire, factuelle, max 8 lignes) :",
            ),
        ]
    )
    


    def format_docs(docs, max_chars=2000):
        return "\n\n".join(d.page_content for d in docs)[:max_chars]


    rag_chain = (
        RunnableParallel(
        {
            "question": RunnablePassthrough(),# On garde la question brute
            "context": (lambda x: x["question"]) | retriever | format_docs,# On cherche dans FAISS
            "user_context": RunnablePassthrough(),# On garde le document uploadé
        }
    )
    | prompt # On injecte tout dans le moule
    | llm # Le modèle génère la réponse
    | StrOutputParser() # On transforme la sortie en texte simple


    )

    return rag_chain
