# app/router.py

from typing import Dict
import re

from .agents import create_agents, AgentResponse


class AgentRouter:
    def __init__(self):
        self.agents = create_agents()


    def route(self, question: str) -> str:
        q = question.lower()

        # admissions
        if any(w in q for w in [
        "admission", "candidature", "postuler", "inscription",
        "prérequis", "conditions", "dossier"
        ]):
            return "admissions"

        # vie étudiante
        if any(w in q for w in [
        "club", "bde", "association", "associations",
        "campus", "événement", "event", "soirée", "international"
        ]):
            return "student_life"

        # académique
        if any(w in q for w in [
            "formation", "majeure", "msc", "bachelor",
            "cours", "matière", "ects", "crédits",
            "examen", "partiel", "projet"
            ]   ):
            return "academics"

        # PAS de fallback flou
        return "admin"


        

    def handle(self, question: str, history=None) -> AgentResponse:
        agent_key = self.route(question)
        agent = self.agents[agent_key]
        return agent.run(question, history=history)
