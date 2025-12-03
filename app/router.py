# app/router.py

from typing import Dict
import re

from .agents import create_agents, AgentResponse


class AgentRouter:
    def __init__(self):
        self.agents = create_agents()

    def route(self, question: str) -> str:
        q = question.lower()

        # Très simple pour commencer : tu pourras raffiner
        if any(w in q for w in ["club", "bureau des élèves", "bde", "associations", "event", "soirée", "campus"]):
            return "student_life"

        if any(w in q for w in ["cours", "majeure", "matière", "crédits", "ects", "partiel", "examen", "projet"]):
            return "academics"

        if any(w in q for w in ["inscription", "administratif", "certificat", "absence", "justificatif", "règlement"]):
            return "admin"

        # fallback -> académiques par défaut
        return "academics"

    def handle(self, question: str, history=None) -> AgentResponse:
        agent_key = self.route(question)
        agent = self.agents[agent_key]
        return agent.run(question, history=history)
