# app/agents.py

from dataclasses import dataclass
from typing import List, Dict

from .rag import build_rag_chain


@dataclass
class AgentResponse:
    agent_name: str
    answer: str


class BaseAgent:
    def __init__(self, name: str, system_prompt: str, agent_type: str):
        self.name = name
        self.system_prompt = system_prompt
        self.agent_type = agent_type

        # RAG chain (FAISS + LLM)
        self.chain = build_rag_chain(
            system_prompt=system_prompt,
            agent_type=agent_type,
        )

    def run(self, question: str, history: List[Dict] = None) -> AgentResponse:
        user_context = ""

        if history:
            for msg in reversed(history):
                if msg.get("uploaded_doc"):
                    user_context = msg["uploaded_doc"]
                    break

        result = self.chain.invoke(
            {
                "question": question,                 #  FAISS embedde SEULEMENT la question
                "user_context": user_context[:2000],  #  PDF injecté UNIQUEMENT dans le prompt
            }
        )

        return AgentResponse(
            agent_name=self.name,
            answer=result,
        )




def create_agents():
    agents = {}
    # PROMPT COMMUN 
    common_rules = (
        "Règles de comportement :\n\n"
        " 1 Cas GÉNÉRAL (hors ESILV )\n"
        "- Si la question est une salutation, une discussion générale, une réflexion personnelle "
        "ou une question sur la vie en général, tu peux répondre librement, de manière naturelle, "
        "courte et bienveillante.\n\n"
        "2 Cas INSTITUTIONNEL (ESILV)\n"
        "- Dès que la question concerne l’ESILV, ou leurs formations, admissions, "
        "services, règles ou fonctionnement :\n"
        "  - Tu réponds UNIQUEMENT à partir des documents fournis par le système (RAG).\n"
        "  - Tu n’utilises AUCUNE connaissance externe.\n"
        "  - Tu n’inventes JAMAIS d’information.\n"
        "  - Si l’information n’est pas clairement présente dans les documents, réponds EXACTEMENT :\n"
        "    \"Je ne peux pas fournir les informations nécessaires à partir de mes sources. "
        "Veuillez remplir le formulaire de contact pour obtenir une réponse. "
        "Avez-vous une autre demande ?\"\n\n"
        "Style de réponse :\n"
        "- Clair, concis, poli.\n"
        "- Pas de blabla inutile.\n"
        "- Ton professionnel mais humain.\n"
    )

    # AGENT VIE ÉTUDIANTE
    student_life_prompt = (
        "Tu es l’agent officiel VIE ÉTUDIANTE.\n"
        "Périmètre STRICT : associations, BDE, événements, campus, vie étudiante.\n\n"
        "Hors périmètre institutionnel → applique les règles générales.\n\n"
        + common_rules
    )

    # AGENT ACADÉMIQUE
    academics_prompt = (
        "Tu es l’agent officiel ACADÉMIQUE.\n"
        "Périmètre STRICT : formations, programmes, cours, majeures, ECTS, projets.\n\n"
        "Hors périmètre institutionnel → applique les règles générales.\n\n"
        + common_rules
    )

    # AGENT ADMISSIONS
    admissions_prompt = (
        "Tu es l’agent officiel ADMISSIONS.\n"
        "Périmètre STRICT : admission, candidature, prérequis, calendrier.\n\n"
        "RÈGLE OBLIGATOIRE :\n"
        "À la fin de TOUTE réponse liée aux admissions, invite à remplir le formulaire.\n\n"
        "Phrase attendue :\n"
        "\"Pour aller plus loin ou être recontacté, vous pouvez remplir le formulaire de contact "
        "disponible dans la barre latérale gauche.\"\n\n"
        "Hors périmètre institutionnel → applique les règles générales.\n\n"
        + common_rules
    )

    # AGENT ADMINISTRATIF
    admin_prompt = (
        "Tu es l’agent officiel ADMINISTRATIF.\n"
        "Périmètre STRICT : certificats, absences, règlements, documents internes.\n\n"
        "Hors périmètre institutionnel → applique les règles générales.\n\n"
        + common_rules
    )

    # Register agents

    agents["student_life"] = BaseAgent(
        "StudentLifeAgent", student_life_prompt, agent_type="student_life"
    )
    agents["academics"] = BaseAgent(
        "AcademicsAgent", academics_prompt, agent_type="academics"
    )
    agents["admissions"] = BaseAgent(
        "AdmissionsAgent", admissions_prompt, agent_type="admissions"
    )
    agents["admin"] = BaseAgent(
        "AdminAgent", admin_prompt, agent_type="admin"
    )

    return agents
