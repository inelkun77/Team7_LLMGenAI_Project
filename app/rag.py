# app/rag.py

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from .config import OLLAMA_MODEL, EMBEDDING_MODEL, VECTORSTORE_PATH


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # OK en local projet étudiant
    )
    return vectorstore


def build_rag_chain(system_prompt: str):
    """
    Construit une chaîne RAG simple avec LCEL :
    question -> retrieve -> prompt -> LLM -> texte
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question: {question}\n\n"
                "Context (extraits de documents ESILV):\n{context}\n\n"
                "Réponse (en français, claire, structurée) :",
            ),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Pipeline LCEL
    rag_chain = (
        RunnableParallel(
            {
                "question": RunnablePassthrough(),
                "context": retriever | format_docs,
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
