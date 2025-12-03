# app/agents.py

from dataclasses import dataclass
from typing import List, Dict

from .rag import build_rag_chain


@dataclass
class AgentResponse:
    agent_name: str
    answer: str


class BaseAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.chain = build_rag_chain(system_prompt)

    def run(self, question: str, history: List[Dict] = None) -> AgentResponse:
        # On pourrait injecter l'historique dans le prompt si besoin.
        result = self.chain.invoke(question)
        return AgentResponse(agent_name=self.name, answer=result)


def create_agents():
    agents = {}

    student_life_prompt = (
        "Tu es un assistant spécialisé dans la vie étudiante à l'ESILV "
        "(associations, clubs, événements, campus, services aux étudiants, etc.). "
        "Tu connais très bien le contexte de l'école et tu t'appuies sur les "
        "documents fournis. Si une information manque, dis-le honnêtement."
    )
    academics_prompt = (
        "Tu es un assistant spécialisé dans les aspects académiques de l'ESILV "
        "(majeures, cours, projets, évaluations, emploi du temps, crédits ECTS, etc.). "
        "Réponds toujours de manière précise, structurée, et en citant le contexte si utile."
    )
    admin_prompt = (
        "Tu es un assistant spécialisé dans les procédures et aspects administratifs de l'ESILV "
        "(inscriptions, absences, rattrapages, règlements, formalités, etc.). "
        "Tu t'appuies sur les documents ESILV fournis."
    )

    agents["student_life"] = BaseAgent("StudentLifeAgent", student_life_prompt)
    agents["academics"] = BaseAgent("AcademicsAgent", academics_prompt)
    agents["admin"] = BaseAgent("AdminAgent", admin_prompt)

    return agents
