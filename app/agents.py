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
    "### RÈGLES DE COMPORTEMENT CRITIQUES (Priorité Absolue) ###\n\n"
    
    "1. CAS DU 'SMALL TALK' (Salutations / Questions générales) :\n"
    "- Si l'utilisateur dit 'bonjour', 'ça va', ou pose une question de courtoisie : "
    "réponds de manière TRÈS COURTE et polie (ex: 'Bonjour ! Je vais bien, merci. Comment puis-je vous aider ?').\n"
    "- INTERDICTION : Ne mentionne JAMAIS l'ESILV, la cybersécurité ou les formations dans ce cas.\n\n"

    "2. CAS DES DONNÉES PERSONNELLES (Candidatures / Dossiers) :\n"
    "- Tu n'as AUCUN accès aux dossiers nominatifs ni aux résultats des candidats.\n"
    "- Si on te demande 'où en est ma candidature' ou 'ai-je été reçu' :\n"
    "  - Tu as l'INTERDICTION FORMELLE d'inventer une réponse positive ou négative.\n"
    "  - Réponds obligatoirement : 'Je n'ai pas accès à votre dossier personnel. Veuillez consulter votre portail candidat ou contacter le service admission via le formulaire.'\n\n"

    "3. CAS INSTITUTIONNEL (ESILV / Formations) :\n"
    "- Uniquement si la question porte EXPLICITEMENT sur l'école ou ses programmes :\n"
    "  - Réponds UNIQUEMENT à partir des documents fournis (RAG).\n"
    "  - Ne dévie jamais vers des conseils non sollicités (ex: ne propose pas de spécialité si on ne te demande pas de conseil).\n"
    "  - N'utilise AUCUNE connaissance externe.\n\n"

    "4. ÉCHEC ET SÉCURITÉ :\n"
    "- Si l'info n'est pas dans les documents, réponds EXACTEMENT : "
    "'Je ne peux pas fournir les informations nécessaires à partir de mes sources. Veuillez remplir le formulaire de contact.'\n"
    "- Ne jamais inventer de faits (ex: dates de rentrée, processus d'admission imaginaires)."
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
